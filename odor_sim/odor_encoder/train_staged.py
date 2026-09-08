"""Train one multi-task odor Transformer through Phases A, B, and C."""

from __future__ import annotations

import argparse
import json
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
    "B": {"static": 0.4, "virtual": 0.6},
    "C": {"static": 0.2, "virtual": 0.3, "arm": 0.5},
}


class CausalWindows(Dataset):
    def __init__(self, root: Path, split: str, *, rate: float, max_s: float, samples: int, seed: int):
        self.files = sorted((root / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no episodes under {root / split}")
        self.rate, self.max_steps, self.samples, self.seed = rate, int(round(max_s * rate)), samples, seed
        self.pure_only = False
        self.prefix_steps = [int(round(s * rate)) for s in (3, 5, 7, 10) if s <= max_s]
        baseline = [MOXSensor(model=m, rate=rate).baseline_voltage() for m in SENSOR_ORDER]
        self.baseline = np.asarray(baseline, np.float32)

    def __len__(self):
        return self.samples

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + index)
        for _ in range(30):
            data = np.load(self.files[int(rng.integers(len(self.files)))])
            valid = np.flatnonzero(data["label_valid"])
            if self.pure_only:
                pure = np.asarray(data["label_presence"]).sum(1) <= 1
                valid = valid[pure[valid]]
            prefix = int(rng.choice(self.prefix_steps))
            valid = valid[valid >= prefix - 1]
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
        return (
            torch.from_numpy(padded), torch.from_numpy(mask),
            torch.from_numpy(np.asarray(data["label_presence"][end], np.float32)),
            torch.from_numpy(np.asarray(data["label_fractions"][end], np.float32)),
            torch.tensor(float(data["label_intensity"][end]), dtype=torch.float32),
            torch.tensor(float(data["label_clean"][end]), dtype=torch.float32),
            torch.tensor(prefix, dtype=torch.int64),
        )


class OdorTransformer(nn.Module):
    def __init__(self, max_steps: int, dim: int = 256, heads: int = 8, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Linear(len(SENSOR_ORDER), dim)
        self.position = nn.Parameter(torch.randn(1, max_steps, dim) * 0.01)
        block = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(dim)
        self.presence = nn.Linear(dim, 6)
        self.fractions = nn.Linear(dim, 6)
        self.intensity = nn.Linear(dim, 1)
        self.clean = nn.Linear(dim, 1)

    def forward(self, x, padding_mask):
        z = self.encoder(self.input(x) + self.position[:, :x.shape[1]], src_key_padding_mask=padding_mask)
        z = self.norm(z[:, -1])
        return self.presence(z), self.fractions(z), F.softplus(self.intensity(z).squeeze(-1)), self.clean(z).squeeze(-1)


def _loss(outputs, targets):
    presence, fraction_logits, intensity, clean = outputs
    yp, yf, yi, yc = targets
    presence_loss = F.binary_cross_entropy_with_logits(presence, yp)
    nonempty = yf.sum(1) > 0
    fraction_loss = -(yf[nonempty] * F.log_softmax(fraction_logits[nonempty], 1)).sum(1).mean() if nonempty.any() else presence_loss * 0
    return presence_loss + fraction_loss + F.smooth_l1_loss(intensity, yi) + F.binary_cross_entropy_with_logits(clean, yc)


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


@torch.no_grad()
def evaluate(model, dataset: CausalWindows, device, batch_size: int, samples: int) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    totals = {str(s): {"n": 0, "presence_correct": 0, "fraction_abs": 0.0, "intensity_abs": 0.0, "clean_correct": 0} for s in dataset.prefix_steps}
    for batch in loader:
        x, mask, yp, yf, yi, yc, prefix = [item.to(device) for item in batch]
        op, of, oi, oc = model(x, mask)
        pp, pf, pc = (torch.sigmoid(op) >= 0.5), torch.softmax(of, 1), (torch.sigmoid(oc) >= 0.5)
        for value in prefix.unique():
            choose = prefix == value
            bucket = totals[str(int(value))]
            n = int(choose.sum())
            bucket["n"] += n
            bucket["presence_correct"] += int((pp[choose] == yp[choose].bool()).all(1).sum())
            bucket["fraction_abs"] += float((pf[choose] - yf[choose]).abs().mean(1).sum())
            bucket["intensity_abs"] += float((oi[choose] - yi[choose]).abs().sum())
            bucket["clean_correct"] += int((pc[choose] == yc[choose].bool()).sum())
        if sum(v["n"] for v in totals.values()) >= samples:
            break
    return {k: {"n": v["n"], "presence_exact": v["presence_correct"] / max(v["n"], 1),
                "fraction_mae": v["fraction_abs"] / max(v["n"], 1),
                "intensity_logppm_mae": v["intensity_abs"] / max(v["n"], 1),
                "clean_accuracy": v["clean_correct"] / max(v["n"], 1)} for k, v in totals.items()}


def train(args):
    roots = {"static": Path(args.static), "virtual": Path(args.virtual), "arm": Path(args.arm)}
    train_sets = {name: CausalWindows(root, "train", rate=args.rate, max_s=args.max_history_s, samples=args.steps_per_epoch * args.batch_size, seed=args.seed + i * 100000) for i, (name, root) in enumerate(roots.items())}
    val_sets = {name: CausalWindows(root, "validation", rate=args.rate, max_s=args.max_history_s, samples=args.eval_samples, seed=args.seed + 500000 + i * 100000) for i, (name, root) in enumerate(roots.items())}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = OdorTransformer(train_sets["static"].max_steps, args.dim, args.heads, args.layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for phase in ("A", "B", "C"):
        loaders = _domain_loaders(train_sets, PHASES[phase], args.batch_size)
        for epoch in range(getattr(args, f"epochs_{phase.lower()}")):
            train_sets["static"].pure_only = phase == "A" and epoch < max(1, int(round(args.epochs_a * 0.10)))
            model.train(); running = 0.0
            for _ in range(args.steps_per_epoch):
                chunks = [next(loaders[name]) for name in PHASES[phase]]
                batch = [torch.cat(items, 0).to(device) for items in zip(*chunks)]
                x, mask, yp, yf, yi, yc, _ = batch
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(model(x, mask), (yp, yf, yi, yc))
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                running += float(loss)
            print(f"[train] phase={phase} epoch={epoch + 1} loss={running / args.steps_per_epoch:.4f}")
        train_sets["static"].pure_only = False
        metrics = {domain: evaluate(model, dataset, device, args.batch_size, args.eval_samples) for domain, dataset in val_sets.items()}
        all_metrics[phase] = metrics
        torch.save({"model_state": model.state_dict(), "phase": phase, "gas_order": GAS_ORDER,
                    "sensor_order": SENSOR_ORDER, "config": vars(args), "validation": metrics}, out / f"phase_{phase.lower()}.pt")
        (out / "metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
    return model


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--static", default="datasets/odor_encoder/static")
    p.add_argument("--virtual", default="datasets/odor_encoder/virtual")
    p.add_argument("--arm", default="datasets/odor_encoder/arm")
    p.add_argument("--out-dir", default="checkpoints/odor_encoder")
    p.add_argument("--epochs-a", type=int, default=20); p.add_argument("--epochs-b", type=int, default=20); p.add_argument("--epochs-c", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=500); p.add_argument("--eval-samples", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--rate", type=float, default=20.0); p.add_argument("--max-history-s", type=float, default=10.0)
    p.add_argument("--dim", type=int, default=256); p.add_argument("--heads", type=int, default=8); p.add_argument("--layers", type=int, default=4); p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--device", choices=("cpu", "cuda")); p.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
