"""Train the A/B ScentFormer odor encoder with static-to-GADEN replay."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from odor_sim.odor_encoder.dataset import GAS_ORDER, SENSOR_ORDER
from odor_sim.sensors.mox_pid import MOXSensor

PHASES = {
    "A": {"static": 1.0},
    "B": {"static": 0.2, "virtual": 0.8},
}

CONCENTRATION_BIN_NAMES = ("<1", "1-10", "10-30", "30-60", "60-100", ">100")
SNIFF_ELAPSED_BIN_NAMES = ("0-1s", "1-3s", "3-5s", "5-10s", ">=10s")


class CausalWindows(Dataset):
    def __init__(self, root: Path, split: str, *, rate: float, max_s: float, samples: int, seed: int, random_sampling: bool = True, fixed_prefix_s: float | None = None, cache_size: int = 256, include_metadata: bool = False, minimum_endpoint_s: float | None = None):
        self.files = sorted((root / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no episodes under {root / split}")
        self.rate, self.max_steps, self.samples, self.seed = rate, int(round(max_s * rate)), samples, seed
        self.pure_only = False
        self.prefix_seconds = [s for s in (1, 3, 5, 7, 10, 20, 30, 40) if s <= max_s]
        self.prefix_steps = [int(round(s * rate)) for s in self.prefix_seconds]
        self.fixed_prefix_steps = None if fixed_prefix_s is None else int(round(fixed_prefix_s * rate))
        self.minimum_endpoint_steps = 0 if minimum_endpoint_s is None else int(round(minimum_endpoint_s * rate))
        self.random_sampling = bool(random_sampling)
        self.include_metadata = bool(include_metadata)
        self.rng = np.random.default_rng(seed)
        self.cache_size = max(0, int(cache_size))
        self.cache = OrderedDict()
        baseline = [MOXSensor(model=m, rate=rate).baseline_voltage() for m in SENSOR_ORDER]
        self.baseline = np.asarray(baseline, np.float32)

    def __len__(self):
        return self.samples

    def _load(self, path: Path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        with np.load(path) as raw:
            data = {key: np.asarray(raw[key]) for key in raw.files}
        if self.cache_size:
            self.cache[path] = data
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return data

    def __getitem__(self, index: int):
        rng = self.rng if self.random_sampling else np.random.default_rng(self.seed + index)
        for _ in range(100):
            data = self._load(self.files[int(rng.integers(len(self.files)))])
            valid = np.flatnonzero(data["label_valid"])
            if self.pure_only:
                pure = np.asarray(data["label_presence"]).sum(1) <= 1
                valid = valid[pure[valid]]
            prefix = self.fixed_prefix_steps or int(rng.choice(self.prefix_steps))
            required = max(prefix, self.minimum_endpoint_steps)
            valid = valid[valid >= required - 1]
            if len(valid):
                end = int(rng.choice(valid))
                break
        else:
            raise RuntimeError("could not find a valid causal window")
        start = end - prefix + 1
        x = (np.asarray(data["voltage"][start:end + 1], np.float32) - self.baseline) / np.maximum(self.baseline, 1e-4)
        padded = np.zeros((self.max_steps, len(SENSOR_ORDER)), np.float32)
        mask = np.ones(self.max_steps, bool)
        padded[-prefix:] = x
        mask[-prefix:] = False
        result = (
            torch.from_numpy(padded), torch.from_numpy(mask),
            torch.from_numpy(np.asarray(data["label_presence"][end], np.float32)),
            torch.from_numpy(np.asarray(data["label_fractions"][end], np.float32)),
            torch.tensor(float(data["label_intensity"][end]), dtype=torch.float32),
            torch.tensor(float(data["label_clean"][end]), dtype=torch.float32),
            torch.tensor(float(data["label_recovery"][end]), dtype=torch.float32),
            torch.tensor(prefix, dtype=torch.int64),
        )
        if self.include_metadata:
            if "endpoint_kind" in data:
                endpoint_kind = str(data["endpoint_kind"][end])
            else:
                endpoint_kind = "recovery" if data["label_recovery"][end] > 0.5 else "controlled"
            sniff_elapsed = (
                float(data["sniff_elapsed_s"][end])
                if "sniff_elapsed_s" in data and np.isfinite(data["sniff_elapsed_s"][end])
                else -1.0
            )
            return (
                *result, str(data["phase"][end]), endpoint_kind,
                torch.tensor(sniff_elapsed, dtype=torch.float32),
            )
        return result


class OdorTransformer(nn.Module):
    def __init__(self, max_steps: int, dim: int = 256, heads: int = 8, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Linear(len(SENSOR_ORDER), dim)
        self.scent_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.position = nn.Parameter(torch.randn(1, max_steps + 1, dim) * 0.01)
        block = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(dim)
        self.presence = nn.Linear(dim, 6)
        self.fractions = nn.Linear(dim, 6)
        self.intensity = nn.Linear(dim, 1)
        self.clean = nn.Linear(dim, 1)
        self.recovery = nn.Linear(dim, 1)

    def forward(self, x, padding_mask):
        temporal = self.input(x)
        scent = self.scent_token.expand(x.shape[0], -1, -1)
        z = torch.cat((temporal, scent), dim=1)
        z = z + self.position[:, :z.shape[1]]
        scent_is_valid = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device)
        padding_mask = torch.cat((padding_mask, scent_is_valid), dim=1)
        z = self.encoder(z, src_key_padding_mask=padding_mask)
        z = self.norm(z[:, -1])
        return (
            self.presence(z), self.fractions(z),
            F.softplus(self.intensity(z).squeeze(-1)),
            self.clean(z).squeeze(-1), self.recovery(z).squeeze(-1),
        )


def _loss(outputs, targets, gas_class_weights=None) -> dict[str, torch.Tensor]:
    presence, fraction_logits, intensity, clean, recovery = outputs
    yp, yf, yi, yc, yr = targets
    presence_loss = F.binary_cross_entropy_with_logits(presence, yp)
    nonempty = yf.sum(1) > 0
    if nonempty.any():
        per_sample = -(yf[nonempty] * F.log_softmax(fraction_logits[nonempty], 1)).sum(1)
        if gas_class_weights is not None:
            sample_weights = (yf[nonempty] * gas_class_weights).sum(1)
            fraction_loss = (per_sample * sample_weights).sum() / sample_weights.sum()
        else:
            fraction_loss = per_sample.mean()
    else:
        fraction_loss = presence_loss * 0
    losses = {
        "presence": presence_loss,
        "primary_gas": fraction_loss,
        "intensity": F.smooth_l1_loss(intensity, yi),
        "clean": F.binary_cross_entropy_with_logits(clean, yc),
        "recovery": F.binary_cross_entropy_with_logits(recovery, yr),
    }
    losses["total"] = sum(losses.values())
    return losses


def _new_totals() -> dict:
    return {
        "n": 0, "nonclean_n": 0, "presence_correct": 0,
        "gas_correct": 0, "clean_correct": 0, "fraction_abs": 0.0,
        "intensity_abs": 0.0, "loss_batches": 0,
        "loss_total": 0.0, "loss_presence": 0.0, "loss_primary_gas": 0.0,
        "loss_intensity": 0.0, "loss_clean": 0.0, "loss_recovery": 0.0,
        "gas_confusion": np.zeros((len(GAS_ORDER), len(GAS_ORDER)), np.int64),
        "clean_tp": 0, "clean_tn": 0, "clean_fp": 0, "clean_fn": 0,
        "recovery_tp": 0, "recovery_tn": 0,
        "recovery_fp": 0, "recovery_fn": 0,
        "concentration_count": np.zeros(len(CONCENTRATION_BIN_NAMES), np.int64),
        "concentration_correct": np.zeros(len(CONCENTRATION_BIN_NAMES), np.int64),
        "path_count": {}, "path_correct": {},
        "state_count": {}, "state_correct": {},
        "sniff_count": np.zeros(len(SNIFF_ELAPSED_BIN_NAMES), np.int64),
        "sniff_correct": np.zeros(len(SNIFF_ELAPSED_BIN_NAMES), np.int64),
    }


def _accumulate(
    totals, outputs, targets, losses, paths=None, states=None, sniff_elapsed=None,
) -> None:
    presence, fraction_logits, intensity, clean, recovery = (
        value.detach() for value in outputs
    )
    yp, yf, yi, yc, yr = targets
    nonclean = yf.sum(1) > 0
    n = int(len(yp))
    totals["n"] += n
    totals["nonclean_n"] += int(nonclean.sum())
    totals["presence_correct"] += int(((torch.sigmoid(presence) >= 0.5) == yp.bool()).all(1).sum())
    if nonclean.any():
        predicted_gas = fraction_logits[nonclean].argmax(1)
        true_gas = yf[nonclean].argmax(1)
        gas_correct = predicted_gas == true_gas
        correctness = gas_correct.detach().cpu().tolist()
        totals["gas_correct"] += int(gas_correct.sum())
        encoded = true_gas * len(GAS_ORDER) + predicted_gas
        confusion = torch.bincount(encoded, minlength=len(GAS_ORDER) ** 2)
        totals["gas_confusion"] += confusion.reshape(len(GAS_ORDER), len(GAS_ORDER)).cpu().numpy()

        concentration = torch.expm1(yi[nonclean])
        boundaries = torch.tensor((1.0, 10.0, 30.0, 60.0, 100.0), device=yi.device)
        bins = torch.bucketize(concentration, boundaries)
        totals["concentration_count"] += torch.bincount(
            bins, minlength=len(CONCENTRATION_BIN_NAMES)
        ).cpu().numpy()
        totals["concentration_correct"] += torch.bincount(
            bins, weights=gas_correct.float(), minlength=len(CONCENTRATION_BIN_NAMES)
        ).to(torch.int64).cpu().numpy()
        if paths is not None:
            nonclean_paths = [path for path, keep in zip(paths, nonclean.cpu().tolist()) if keep]
            for path, correct in zip(nonclean_paths, correctness):
                totals["path_count"][path] = totals["path_count"].get(path, 0) + 1
                totals["path_correct"][path] = totals["path_correct"].get(path, 0) + int(correct)
        if states is not None:
            nonclean_states = [state for state, keep in zip(states, nonclean.cpu().tolist()) if keep]
            for state, correct in zip(nonclean_states, correctness):
                totals["state_count"][state] = totals["state_count"].get(state, 0) + 1
                totals["state_correct"][state] = totals["state_correct"].get(state, 0) + int(correct)
        if sniff_elapsed is not None:
            elapsed = sniff_elapsed[nonclean]
            usable = elapsed >= 0
            if usable.any():
                elapsed_bins = torch.bucketize(
                    elapsed[usable], torch.tensor((1.0, 3.0, 5.0, 10.0), device=yi.device),
                )
                elapsed_correct = gas_correct[usable]
                totals["sniff_count"] += torch.bincount(
                    elapsed_bins, minlength=len(SNIFF_ELAPSED_BIN_NAMES),
                ).cpu().numpy()
                totals["sniff_correct"] += torch.bincount(
                    elapsed_bins, weights=elapsed_correct.float(),
                    minlength=len(SNIFF_ELAPSED_BIN_NAMES),
                ).to(torch.int64).cpu().numpy()

    predicted_clean = torch.sigmoid(clean) >= 0.5
    true_clean = yc.bool()
    totals["clean_correct"] += int((predicted_clean == true_clean).sum())
    totals["clean_tp"] += int((predicted_clean & true_clean).sum())
    totals["clean_tn"] += int((~predicted_clean & ~true_clean).sum())
    totals["clean_fp"] += int((predicted_clean & ~true_clean).sum())
    totals["clean_fn"] += int((~predicted_clean & true_clean).sum())
    predicted_recovery = torch.sigmoid(recovery) >= 0.5
    true_recovery = yr.bool()
    totals["recovery_tp"] += int((predicted_recovery & true_recovery).sum())
    totals["recovery_tn"] += int((~predicted_recovery & ~true_recovery).sum())
    totals["recovery_fp"] += int((predicted_recovery & ~true_recovery).sum())
    totals["recovery_fn"] += int((~predicted_recovery & true_recovery).sum())
    if nonclean.any():
        totals["fraction_abs"] += float(
            (torch.softmax(fraction_logits[nonclean], 1) - yf[nonclean]).abs().mean(1).sum()
        )
    totals["intensity_abs"] += float((intensity - yi).abs().sum())
    totals["loss_batches"] += 1
    for name, value in losses.items():
        totals[f"loss_{name}"] += float(value.detach())


def _finalize(totals) -> dict:
    n = max(int(totals["n"]), 1)
    nonclean_n = max(int(totals["nonclean_n"]), 1)
    loss_batches = max(int(totals["loss_batches"]), 1)
    confusion = np.asarray(totals["gas_confusion"])
    per_gas_count = confusion.sum(1)
    per_gas_accuracy = np.divide(
        confusion.diagonal(), per_gas_count,
        out=np.zeros(len(GAS_ORDER), np.float64), where=per_gas_count > 0,
    )
    clean_recall = totals["clean_tp"] / max(totals["clean_tp"] + totals["clean_fn"], 1)
    nonclean_recall = totals["clean_tn"] / max(totals["clean_tn"] + totals["clean_fp"], 1)
    concentration_accuracy = np.divide(
        totals["concentration_correct"], totals["concentration_count"],
        out=np.zeros(len(CONCENTRATION_BIN_NAMES), np.float64),
        where=totals["concentration_count"] > 0,
    )
    sniff_accuracy = np.divide(
        totals["sniff_correct"], totals["sniff_count"],
        out=np.zeros(len(SNIFF_ELAPSED_BIN_NAMES), np.float64),
        where=totals["sniff_count"] > 0,
    )
    return {
        "loss": {
            name: totals[f"loss_{name}"] / loss_batches
            for name in ("total", "presence", "primary_gas", "intensity", "clean", "recovery")
        },
        "metrics": {
            "n": int(totals["n"]),
            "nonclean_n": int(totals["nonclean_n"]),
            "presence_exact": totals["presence_correct"] / n,
            "primary_gas_accuracy_nonclean": totals["gas_correct"] / nonclean_n,
            "primary_gas_probability_mae_nonclean": totals["fraction_abs"] / nonclean_n,
            "intensity_logppm_mae": totals["intensity_abs"] / n,
            "clean_accuracy": totals["clean_correct"] / n,
            "clean_precision": totals["clean_tp"] / max(totals["clean_tp"] + totals["clean_fp"], 1),
            "clean_recall": clean_recall,
            "nonclean_recall": nonclean_recall,
            "clean_balanced_accuracy": 0.5 * (clean_recall + nonclean_recall),
            "clean_counts": {
                "tp": totals["clean_tp"], "tn": totals["clean_tn"],
                "fp": totals["clean_fp"], "fn": totals["clean_fn"],
            },
            "recovery_precision": totals["recovery_tp"] / max(totals["recovery_tp"] + totals["recovery_fp"], 1),
            "recovery_recall": totals["recovery_tp"] / max(totals["recovery_tp"] + totals["recovery_fn"], 1),
            "recovery_balanced_accuracy": 0.5 * (
                totals["recovery_tp"] / max(totals["recovery_tp"] + totals["recovery_fn"], 1)
                + totals["recovery_tn"] / max(totals["recovery_tn"] + totals["recovery_fp"], 1)
            ),
            "recovery_counts": {
                "tp": totals["recovery_tp"], "tn": totals["recovery_tn"],
                "fp": totals["recovery_fp"], "fn": totals["recovery_fn"],
            },
            "gas_confusion_matrix": confusion.tolist(),
            "per_gas_accuracy": {
                gas: {"n": int(per_gas_count[i]), "accuracy": float(per_gas_accuracy[i])}
                for i, gas in enumerate(GAS_ORDER)
            },
            "macro_gas_accuracy": float(per_gas_accuracy[per_gas_count > 0].mean()) if np.any(per_gas_count) else 0.0,
            "gas_accuracy_by_total_ppm": {
                name: {"n": int(totals["concentration_count"][i]), "accuracy": float(concentration_accuracy[i])}
                for i, name in enumerate(CONCENTRATION_BIN_NAMES)
            },
            "gas_accuracy_by_path": {
                path: {
                    "n": int(count),
                    "accuracy": totals["path_correct"].get(path, 0) / max(count, 1),
                }
                for path, count in sorted(totals["path_count"].items())
            },
            "gas_accuracy_by_trajectory_phase": {
                path: {
                    "n": int(count),
                    "accuracy": totals["path_correct"].get(path, 0) / max(count, 1),
                }
                for path, count in sorted(totals["path_count"].items())
            },
            "gas_accuracy_by_exposure_state": {
                state: {
                    "n": int(count),
                    "accuracy": totals["state_correct"].get(state, 0) / max(count, 1),
                }
                for state, count in sorted(totals["state_count"].items())
            },
            "gas_accuracy_by_sniff_elapsed": {
                name: {
                    "n": int(totals["sniff_count"][i]),
                    "accuracy": float(sniff_accuracy[i]),
                }
                for i, name in enumerate(SNIFF_ELAPSED_BIN_NAMES)
            },
        },
    }


def _forever(loader):
    while True:
        yield from loader


def _domain_loaders(datasets: dict[str, CausalWindows], weights: dict[str, float], batch_size: int):
    loaders = {}
    remaining = batch_size
    names = list(weights)
    for i, name in enumerate(names):
        size = remaining if i == len(names) - 1 else max(1, int(round(batch_size * weights[name])))
        remaining -= size
        loaders[name] = _forever(DataLoader(datasets[name], batch_size=size, shuffle=False, drop_last=True))
    return loaders


def _gas_class_weights(args, device) -> torch.Tensor:
    return torch.as_tensor(args.gas_class_weights, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, root: Path, device, args, split: str) -> dict:
    model.eval()
    results = {}
    class_weights = _gas_class_weights(args, device)
    for prefix_s in (3, 5, 7, 10, 15, 20):
        if prefix_s > args.max_history_s:
            continue
        dataset = CausalWindows(
            root, split, rate=args.rate, max_s=args.max_history_s,
            samples=args.eval_samples, seed=args.seed + 900000,
            random_sampling=False, fixed_prefix_s=prefix_s, include_metadata=True,
            minimum_endpoint_s=args.max_history_s,
        )
        totals = _new_totals()
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
            *tensor_items, paths, states, sniff_elapsed = batch
            x, mask, yp, yf, yi, yc, yr, _ = [item.to(device) for item in tensor_items]
            sniff_elapsed = sniff_elapsed.to(device)
            outputs = model(x, mask)
            targets = (yp, yf, yi, yc, yr)
            losses = _loss(outputs, targets, class_weights)
            _accumulate(totals, outputs, targets, losses, paths, states, sniff_elapsed)
        results[f"{prefix_s}s"] = _finalize(totals)
    return results


@torch.no_grad()
def evaluate_epoch(model, root: Path, device, args, seed: int) -> dict:
    """Evaluate one fixed mixed-prefix validation sample after an epoch."""
    model.eval()
    dataset = CausalWindows(
        root, "validation", rate=args.rate, max_s=args.max_history_s,
        samples=args.epoch_eval_samples, seed=seed, random_sampling=False,
    )
    totals = _new_totals()
    class_weights = _gas_class_weights(args, device)
    for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
        x, mask, yp, yf, yi, yc, yr, _ = [item.to(device) for item in batch]
        outputs = model(x, mask)
        targets = (yp, yf, yi, yc, yr)
        losses = _loss(outputs, targets, class_weights)
        _accumulate(totals, outputs, targets, losses)
    return _finalize(totals)


def _save_checkpoint(path, model, phase, epoch, args, architecture, metrics) -> None:
    torch.save(
        {
            "model_state": model.state_dict(), "phase": phase, "epoch": epoch,
            "gas_order": GAS_ORDER, "sensor_order": SENSOR_ORDER,
            "config": vars(args), "architecture": architecture, "metrics": metrics,
        },
        path,
    )


def evaluate_checkpoint(args) -> dict:
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required with --evaluate")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    architecture = checkpoint["architecture"]
    for name in (
        "rate", "max_history_s", "dim", "heads", "layers", "dropout",
        "gas_class_weights",
    ):
        setattr(args, name, config[name])
    model = OdorTransformer(
        int(round(args.max_history_s * args.rate)), args.dim, args.heads,
        args.layers, args.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    roots = (
        {"virtual": Path(args.virtual)}
        if config.get("training_phases") == "B"
        else {"static": Path(args.static), "virtual": Path(args.virtual)}
    )
    results = {
        "checkpoint": str(args.checkpoint), "split": args.evaluate,
        "gas_order": list(GAS_ORDER),
        "domains": {
            domain: evaluate(model, root, device, args, args.evaluate)
            for domain, root in roots.items()
        },
    }
    destination = Path(args.eval_output) if args.eval_output else Path(args.out_dir) / f"{args.evaluate}_metrics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[evaluate] checkpoint={args.checkpoint} split={args.evaluate} output={destination}")
    return results


def train(args):
    roots = (
        {"virtual": Path(args.virtual)}
        if args.training_phases == "B"
        else {"static": Path(args.static), "virtual": Path(args.virtual)}
    )
    train_sets = {name: CausalWindows(root, "train", rate=args.rate, max_s=args.max_history_s, samples=args.steps_per_epoch * args.batch_size, seed=args.seed + i * 100000) for i, (name, root) in enumerate(roots.items())}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    reference_set = next(iter(train_sets.values()))
    model = OdorTransformer(reference_set.max_steps, args.dim, args.heads, args.layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    class_weights = _gas_class_weights(args, device)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    history = {
        "schema_version": 2, "status": "running", "config": vars(args),
        "epochs": [],
    }
    architecture = {
        "name": "ScentFormerAB",
        "pooling": "learned appended SCENT token",
        "input_channels": list(SENSOR_ORDER),
        "normalization": "(voltage - clean_air_baseline) / clean_air_baseline",
        "clean_air_baseline_v": reference_set.baseline.tolist(),
        "gas_order": list(GAS_ORDER),
        "model_dim": args.dim,
        "heads": args.heads,
        "layers": args.layers,
        "max_history_s": args.max_history_s,
        "prefixes_s": [3, 5, 7, 10, 15, 20],
        "target_semantics": "one primary gas; trace background gases are input nuisance only",
        "outputs": [
            "gas_presence_6", "primary_gas_distribution_6", "log_total_ppm_1",
            "baseline_ready_clean_air_1", "post_exposure_recovery_1",
        ],
        "gas_class_weights": dict(zip(GAS_ORDER, args.gas_class_weights)),
        "evaluation_sampling": "paired endpoints across all prefix durations",
        "phase_batch_weights": (
            {"B": {"virtual": 1.0}} if args.training_phases == "B" else PHASES
        ),
    }
    (out / "architecture.json").write_text(json.dumps(architecture, indent=2) + "\n")
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    print(f"[train-ab] device={device} architecture={architecture}")
    phases = ("B",) if args.training_phases == "B" else ("A", "B")
    for phase in phases:
        best_loss = float("inf")
        best_accuracy = -1.0
        selection_domain = "static" if phase == "A" else "virtual"
        weights = {"virtual": 1.0} if args.training_phases == "B" else PHASES[phase]
        loaders = _domain_loaders(train_sets, weights, args.batch_size)
        for epoch in range(getattr(args, f"epochs_{phase.lower()}")):
            model.train()
            train_totals = _new_totals()
            for _ in range(args.steps_per_epoch):
                chunks = [next(loaders[name]) for name in weights]
                batch = [torch.cat(items, 0).to(device) for items in zip(*chunks)]
                x, mask, yp, yf, yi, yc, yr, _ = batch
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    outputs = model(x, mask)
                    targets = (yp, yf, yi, yc, yr)
                    losses = _loss(outputs, targets, class_weights)
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                _accumulate(train_totals, outputs, targets, losses)
            train_summary = _finalize(train_totals)
            validation = {
                domain: evaluate_epoch(model, root, device, args, args.seed + 700000 + i * 10000)
                for i, (domain, root) in enumerate(roots.items())
            }
            epoch_record = {
                "phase": phase, "epoch": epoch + 1,
                "global_epoch": len(history["epochs"]) + 1,
                "train": train_summary, "validation": validation,
            }
            history["epochs"].append(epoch_record)
            (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            selection = validation[selection_domain]
            if selection["loss"]["total"] < best_loss:
                best_loss = selection["loss"]["total"]
                _save_checkpoint(
                    out / f"phase_{phase.lower()}_best_loss.pt", model, phase,
                    epoch + 1, args, architecture, {"epoch_validation": validation},
                )
            accuracy = selection["metrics"]["macro_gas_accuracy"]
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                _save_checkpoint(
                    out / f"phase_{phase.lower()}_best_accuracy.pt", model, phase,
                    epoch + 1, args, architecture, {"epoch_validation": validation},
                )
            losses_text = " ".join(f"{name}={value:.4f}" for name, value in train_summary["loss"].items())
            metrics = train_summary["metrics"]
            print(
                f"[train] phase={phase} epoch={epoch + 1} {losses_text} "
                f"gas_acc={metrics['primary_gas_accuracy_nonclean']:.4f} "
                f"macro_gas_acc={metrics['macro_gas_accuracy']:.4f} "
                f"presence_exact={metrics['presence_exact']:.4f} "
                f"clean_bal_acc={metrics['clean_balanced_accuracy']:.4f} "
                f"recovery_bal_acc={metrics['recovery_balanced_accuracy']:.4f}"
            )
        if "static" in train_sets:
            train_sets["static"].pure_only = False
        metrics = {domain: evaluate(model, root, device, args, "validation") for domain, root in roots.items()}
        all_metrics[phase] = {"validation": metrics}
        _save_checkpoint(
            out / f"phase_{phase.lower()}.pt", model, phase,
            getattr(args, f"epochs_{phase.lower()}"), args, architecture,
            all_metrics[phase],
        )
        (out / "metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
    history["status"] = "complete"
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return model


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--static", default="datasets/odor_encoder/static_single_v3")
    p.add_argument("--virtual", default="datasets/odor_encoder/virtual_single_v6")
    p.add_argument("--out-dir", default="checkpoints/odor_encoder_ab_v3")
    p.add_argument("--training-phases", choices=("AB", "B"), default="AB")
    p.add_argument("--epochs-a", type=int, default=10); p.add_argument("--epochs-b", type=int, default=15)
    p.add_argument("--steps-per-epoch", type=int, default=250); p.add_argument("--eval-samples", type=int, default=500)
    p.add_argument("--epoch-eval-samples", type=int, default=500, help="Validation windows per domain after every epoch")
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--rate", type=float, default=20.0); p.add_argument("--max-history-s", type=float, default=20.0)
    p.add_argument("--dim", type=int, default=256); p.add_argument("--heads", type=int, default=8); p.add_argument("--layers", type=int, default=4); p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--device", choices=("cpu", "cuda")); p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--gas-class-weights", type=float, nargs=len(GAS_ORDER),
        default=(1.0, 1.0, 1.0, 1.0, 1.2, 1.4),
        metavar=tuple(f"W{i + 1}" for i in range(len(GAS_ORDER))),
        help="primary-gas loss weights in GAS_ORDER; emphasizes the confused ethyl-acetate/toluene pair",
    )
    p.add_argument("--evaluate", choices=("validation", "test"), help="Evaluate a saved checkpoint instead of training")
    p.add_argument("--checkpoint", help="Checkpoint used with --evaluate")
    p.add_argument("--eval-output", help="Optional JSON destination for --evaluate")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if any(weight <= 0 for weight in args.gas_class_weights):
        raise SystemExit("--gas-class-weights values must all be positive")
    if args.evaluate:
        evaluate_checkpoint(args)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
