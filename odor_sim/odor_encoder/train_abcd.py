"""Train and evaluate deployment-consistent four-head ScentFormer on physical A/B/C/D."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from odor_sim.odor_encoder.dataset import GAS_ORDER, SENSOR_ORDER
from odor_sim.sensors.mox_pid import MOXSensor

PHASE_MIX = {
    "A": {"A": 1.0},
    "B": {"A": 0.40, "B": 0.60},
    "C": {"A": 0.25, "B": 0.25, "C": 0.50},
    "D": {"A": 0.20, "B": 0.25, "C": 0.20, "D": 0.35},
}
PREFIXES = {
    "baseline": (1, 3, 5, 7, 10, 20),
    "dual_timescale": (1, 3, 5, 7, 10, 20, 30, 40),
}
TIME_BINS = (
    ("0-0.5s", 0.0, 0.5), ("0.5-1s", 0.5, 1.0), ("1-2s", 1.0, 2.0),
    ("2-3s", 2.0, 3.0), ("3-5s", 3.0, 5.0), ("5-7s", 5.0, 7.0),
    ("7-10s", 7.0, 10.0), (">10s", 10.0, float("inf")),
)
PPM_BINS = (
    ("<1", -float("inf"), 1), ("1-5", 1, 5), ("5-10", 5, 10),
    ("10-30", 10, 30), ("30-60", 30, 60), ("60-100", 60, 100),
    ("100-150", 100, 150), (">150", 150, float("inf")),
)
DOMINANCE_BINS = (
    (">=95%", 0.95, float("inf")), ("90-95%", 0.90, 0.95),
    ("80-90%", 0.80, 0.90), ("70-80%", 0.70, 0.80),
    ("<70%", -float("inf"), 0.70),
)
SECOND_OFFSETS = ("0.5s", "1s", "2s", "3s", "5s", "10s")


class PhysicalWindows(Dataset):
    """Balanced event-endpoint sampling from complete continuous trajectories."""

    def __init__(
        self,
        root: Path,
        split: str,
        *,
        model_type: str,
        rate: float,
        samples: int,
        seed: int,
        fixed_prefix_s: float | None = None,
        random_sampling: bool = True,
        cache_size: int = 96,
        phase: str | None = None,
        presence_threshold_ppm: float = 5.0,
        presence_negative_sample_fraction: float = 0.20,
        gas_supervised_sample_fraction: float = 0.35,
    ):
        self.files = sorted((Path(root) / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no recordings under {Path(root) / split}")
        self.model_type = model_type
        self.rate = float(rate)
        self.samples = int(samples)
        self.seed = int(seed)
        self.fixed_prefix_s = fixed_prefix_s
        self.random_sampling = bool(random_sampling)
        self.prefixes = PREFIXES[model_type]
        self.cache_size = max(0, int(cache_size))
        if phase is None:
            first_sidecar = self.files[0].with_suffix(".json")
            metadata = json.loads(first_sidecar.read_text()) if first_sidecar.is_file() else {}
            domain = str(metadata.get("domain", ""))
            phase = domain.rsplit("_", 1)[-1].upper() if domain else None
        self.phase = phase
        self.presence_threshold_ppm = float(presence_threshold_ppm)
        self.presence_negative_sample_fraction = float(presence_negative_sample_fraction)
        self.gas_supervised_sample_fraction = float(gas_supervised_sample_fraction)
        if self.presence_negative_sample_fraction + self.gas_supervised_sample_fraction > 1.0:
            raise ValueError("presence-negative and gas-supervised sample fractions must sum to <= 1")
        self.recovery_applicable = self.phase != "A"
        self.cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self.sidecars: dict[Path, dict] = {}
        self.by_bucket: dict[str, list[tuple[Path, int]]] = {}
        self.presence_negative: list[tuple[Path, int]] = []
        self.gas_supervised: dict[int, list[tuple[Path, int]]] = {}
        self.entries: list[tuple[Path, int]] = []
        for path in self.files:
            with np.load(path) as data:
                required = {
                    "endpoint_eligible", "endpoint_bucket", "gas_loss_mask",
                    "label_presence", "label_recovery", "label_intensity", "label_gas",
                    "local_total_ppm",
                }
                missing = required.difference(data.files)
                if missing:
                    raise ValueError(f"{path} is not physical schema v4; missing {sorted(missing)}")
                indices = np.flatnonzero(data["endpoint_eligible"])
                buckets = np.asarray(data["endpoint_bucket"])
                local_ppm_values = np.asarray(data["local_total_ppm"])
                gas_masks = np.asarray(data["gas_loss_mask"])
                gas_labels = np.asarray(data["label_gas"])
            for endpoint, bucket in zip(indices, buckets[indices]):
                entry = (path, int(endpoint))
                key = str(bucket)
                self.by_bucket.setdefault(key, []).append(entry)
                self.entries.append(entry)
                local_ppm = float(local_ppm_values[endpoint])
                if local_ppm < self.presence_threshold_ppm:
                    self.presence_negative.append(entry)
                elif bool(gas_masks[endpoint]):
                    gas = int(gas_labels[endpoint])
                    self.gas_supervised.setdefault(gas, []).append(entry)
        if not self.entries:
            raise ValueError(f"no eligible endpoints under {root}/{split}")
        self.bucket_names = sorted(self.by_bucket)
        if not random_sampling and samples > 0 and samples < len(self.entries):
            rng = np.random.default_rng(seed)
            selected = []
            for i in range(samples):
                bucket = self.bucket_names[i % len(self.bucket_names)]
                choices = self.by_bucket[bucket]
                selected.append(choices[int(rng.integers(len(choices)))])
            self.entries = selected
        self.baseline = np.asarray([
            MOXSensor(model=model, rate=rate).baseline_voltage() for model in SENSOR_ORDER
        ], np.float32)
        self.baseline_steps = int(round(20 * rate))
        self.recent_steps = int(round(10 * rate))
        self.history_steps = int(round(40 * rate))
        self.downsample = int(round(rate / 5.0))
        if model_type == "dual_timescale" and not np.isclose(rate / self.downsample, 5.0):
            raise ValueError("dual-timescale input requires a rate divisible to 5 Hz")

    def __len__(self):
        return self.samples if self.random_sampling else len(self.entries)

    def _load(self, path):
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

    def _sidecar(self, path):
        if path not in self.sidecars:
            self.sidecars[path] = json.loads(path.with_suffix(".json").read_text())
        return self.sidecars[path]

    @staticmethod
    def _pad(segment, steps):
        segment = segment[-steps:]
        output = np.zeros((steps, len(SENSOR_ORDER)), np.float32)
        mask = np.ones(steps, bool)
        output[-len(segment):] = segment
        mask[-len(segment):] = False
        return output, mask

    def _dual(self, segment):
        full, full_mask = self._pad(segment, self.history_steps)
        recent = full[-self.recent_steps:]
        recent_mask = full_mask[-self.recent_steps:]
        groups = self.history_steps // self.downsample
        grouped = full.reshape(groups, self.downsample, len(SENSOR_ORDER))
        valid = (~full_mask).reshape(groups, self.downsample)
        count = valid.sum(1)
        history = (grouped * valid[..., None]).sum(1) / np.maximum(count[:, None], 1)
        return recent, recent_mask, history.astype(np.float32), count == 0

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index)
        if self.random_sampling:
            draw = float(rng.random())
            if draw < self.presence_negative_sample_fraction and self.presence_negative:
                choices = self.presence_negative
            elif (
                draw < self.presence_negative_sample_fraction + self.gas_supervised_sample_fraction
                and self.gas_supervised
            ):
                gases = sorted(self.gas_supervised)
                gas = gases[int(rng.integers(len(gases)))]
                choices = self.gas_supervised[gas]
            else:
                bucket = self.bucket_names[int(rng.integers(len(self.bucket_names)))]
                choices = self.by_bucket[bucket]
            path, end = choices[int(rng.integers(len(choices)))]
        else:
            path, end = self.entries[index]
        data = self._load(path)
        prefix_s = float(self.fixed_prefix_s or rng.choice(self.prefixes))
        prefix_steps = int(round(prefix_s * self.rate))
        start = max(0, end - prefix_steps + 1)
        signal = np.asarray(data["processed_mox_8ch"][start:end + 1], np.float32)

        local_total_ppm = float(data["local_total_ppm"][end])
        presence_target = local_total_ppm >= self.presence_threshold_ppm
        gas_mask = bool(data["gas_loss_mask"][end]) and presence_target
        mask_reason = str(data["gas_mask_reason"][end]) if gas_mask else (
            "NO_GAS" if not presence_target else str(data["gas_mask_reason"][end])
        )
        item = {
            "gas_target": torch.tensor(int(data["label_gas"][end]), dtype=torch.long),
            "gas_mask": torch.tensor(float(gas_mask), dtype=torch.float32),
            "presence_target": torch.tensor(float(presence_target), dtype=torch.float32),
            "intensity_target": torch.tensor(float(data["label_intensity"][end]), dtype=torch.float32),
            "recovery_target": torch.tensor(float(data["label_recovery"][end]), dtype=torch.float32),
            "recovery_mask": torch.tensor(float(self.recovery_applicable), dtype=torch.float32),
            "local_total_ppm": torch.tensor(local_total_ppm, dtype=torch.float32),
            "dominant_fraction": torch.tensor(float(data["dominant_fraction"][end]), dtype=torch.float32),
            "sniff_elapsed_s": torch.tensor(
                float(data["sniff_elapsed_s"][end])
                if np.isfinite(data["sniff_elapsed_s"][end]) else -1.0,
                dtype=torch.float32,
            ),
            "phase": str(data["phase"][end]),
            "exposure_state": str(data["exposure_state"][end]),
            "mask_reason": mask_reason,
            "sniff_role": str(data["sniff_role"][end]),
            "endpoint_bucket": str(data["endpoint_bucket"][end]),
            "ordered_pair": str(self._sidecar(path)["ordered_pair"]),
            "episode_id": str(self._sidecar(path)["episode_id"]),
        }
        if self.model_type == "baseline":
            x, mask = self._pad(signal, self.baseline_steps)
            item.update(x=torch.from_numpy(x), padding_mask=torch.from_numpy(mask))
        else:
            recent, recent_mask, history, history_mask = self._dual(signal)
            item.update(
                recent=torch.from_numpy(recent), recent_mask=torch.from_numpy(recent_mask),
                history=torch.from_numpy(history), history_mask=torch.from_numpy(history_mask),
            )
        return item


class TemporalBranch(nn.Module):
    def __init__(self, steps, dim, heads, layers, dropout):
        super().__init__()
        self.input = nn.Linear(len(SENSOR_ORDER), dim)
        self.scent_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.position = nn.Parameter(torch.randn(1, steps + 1, dim) * 0.01)
        block = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask):
        token = self.scent_token.expand(x.shape[0], -1, -1)
        encoded = torch.cat((self.input(x), token), 1)
        encoded = encoded + self.position[:, :encoded.shape[1]]
        token_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device)
        encoded = self.encoder(
            encoded, src_key_padding_mask=torch.cat((mask, token_mask), 1),
        )
        return self.norm(encoded[:, -1])


class BaselineScentFormer(nn.Module):
    """Four prediction heads: gas, presence, local intensity, and recovery."""

    def __init__(self, steps, dim=256, heads=8, layers=4, dropout=0.1):
        super().__init__()
        self.temporal = TemporalBranch(steps, dim, heads, layers, dropout)
        self.gas = nn.Linear(dim, len(GAS_ORDER))
        self.presence = nn.Linear(dim, 1)
        self.intensity = nn.Linear(dim, 1)
        self.recovery = nn.Linear(dim, 1)

    def encode(self, x, mask):
        """Deployment API: return only the odor embedding consumed by OVLA."""
        return self.temporal(x, mask)

    def forward(self, x, mask):
        embedding = self.encode(x, mask)
        return {
            "embedding": embedding,
            "gas_logits": self.gas(embedding),
            "presence": self.presence(embedding).squeeze(-1),
            "intensity": F.softplus(self.intensity(embedding).squeeze(-1)),
            "recovery": self.recovery(embedding).squeeze(-1),
        }


class DualTimescaleScentFormer(nn.Module):
    def __init__(self, steps, dim=256, heads=8, layers=4, dropout=0.1):
        super().__init__()
        self.recent = TemporalBranch(steps, dim, heads, layers, dropout)
        self.history = TemporalBranch(steps, dim, heads, layers, dropout)
        self.fusion = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        self.gas = nn.Linear(dim, len(GAS_ORDER))
        self.presence = nn.Linear(dim, 1)
        self.intensity = nn.Linear(dim, 1)
        self.recovery = nn.Linear(dim, 1)

    def encode(self, recent, recent_mask, history, history_mask):
        """Deployment API: return only the fused odor embedding for OVLA."""
        return self.fusion(torch.cat((
            self.recent(recent, recent_mask),
            self.history(history, history_mask),
        ), 1))

    def forward(self, recent, recent_mask, history, history_mask):
        embedding = self.encode(recent, recent_mask, history, history_mask)
        return {
            "embedding": embedding,
            "gas_logits": self.gas(embedding),
            "presence": self.presence(embedding).squeeze(-1),
            "intensity": F.softplus(self.intensity(embedding).squeeze(-1)),
            "recovery": self.recovery(embedding).squeeze(-1),
        }


def build_model(args):
    if args.model == "baseline":
        return BaselineScentFormer(
            int(round(20 * args.rate)), args.dim, args.heads, args.layers, args.dropout,
        )
    return DualTimescaleScentFormer(
        int(round(10 * args.rate)), args.dim, args.heads, args.layers, args.dropout,
    )


def _to_device(batch, device):
    tensor_keys = (
        "x", "padding_mask", "recent", "recent_mask", "history", "history_mask",
        "gas_target", "gas_mask", "presence_target", "intensity_target",
        "recovery_target", "recovery_mask", "local_total_ppm", "dominant_fraction",
        "sniff_elapsed_s",
    )
    return {
        key: value.to(device) if key in tensor_keys else value
        for key, value in batch.items()
    }


def _forward(model, batch, model_type):
    if model_type == "baseline":
        return model(batch["x"], batch["padding_mask"])
    return model(
        batch["recent"], batch["recent_mask"], batch["history"], batch["history_mask"],
    )


def _loss(outputs, batch, *, lambda_gas=1.0, lambda_presence=0.3,
          lambda_intensity=0.15, lambda_recovery=0.3):
    per_gas = F.cross_entropy(outputs["gas_logits"], batch["gas_target"], reduction="none")
    gas_mask = batch["gas_mask"]
    gas = (per_gas * gas_mask).sum() / gas_mask.sum().clamp_min(1.0)
    per_recovery = F.binary_cross_entropy_with_logits(
        outputs["recovery"], batch["recovery_target"], reduction="none",
    )
    recovery_mask = batch["recovery_mask"]
    recovery = (per_recovery * recovery_mask).sum() / recovery_mask.sum().clamp_min(1.0)
    losses = {
        "gas": gas,
        "presence": F.binary_cross_entropy_with_logits(
            outputs["presence"], batch["presence_target"],
        ),
        "intensity": F.smooth_l1_loss(outputs["intensity"], batch["intensity_target"]),
        "recovery": recovery,
    }
    losses["total"] = (
        lambda_gas * losses["gas"]
        + lambda_presence * losses["presence"]
        + lambda_intensity * losses["intensity"]
        + lambda_recovery * losses["recovery"]
    )
    return losses


def _binary_counts(predicted, target):
    predicted = predicted.bool()
    target = target.bool()
    return {
        "tp": int((predicted & target).sum()), "tn": int((~predicted & ~target).sum()),
        "fp": int((predicted & ~target).sum()), "fn": int((~predicted & target).sum()),
    }


def _binary_metrics(counts):
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    total = tp + tn + fp + fn
    positive_n = tp + fn
    negative_n = tn + fp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    applicable = positive_n > 0 and negative_n > 0
    return {
        "applicable": applicable,
        "positive_n": positive_n, "negative_n": negative_n,
        "positive_fraction": positive_n / max(total, 1),
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "counts": counts,
    }


def _bin_name(value, bins):
    for name, low, high in bins:
        if low <= value < high:
            return name
    return bins[-1][0]


def _new_metrics():
    return {
        "n": 0, "gas_n": 0, "gas_correct": 0, "intensity_abs": 0.0,
        "loss_batches": 0,
        **{f"loss_{name}": 0.0 for name in ("total", "gas", "presence", "intensity", "recovery")},
        "confusion": np.zeros((len(GAS_ORDER), len(GAS_ORDER)), np.int64),
        "presence": Counter(tp=0, tn=0, fp=0, fn=0),
        "recovery": Counter(tp=0, tn=0, fp=0, fn=0),
        "mask_reason": Counter(), "bucket": Counter(),
        "group_n": Counter(), "group_correct": Counter(),
        "intensity_n": Counter(), "intensity_abs_bin": Counter(),
        "confidence": [], "confidence_correct": [], "nll": [], "margin": [],
        "pair_offset_n": Counter(), "pair_offset_correct": Counter(),
        "second_offset_total": Counter(), "second_offset_n": Counter(),
        "second_offset_correct": Counter(),
    }


def _accumulate(metrics, outputs, batch, losses):
    batch_size = len(batch["gas_target"])
    mask = batch["gas_mask"].bool()
    prediction = outputs["gas_logits"].argmax(1)
    correct = prediction == batch["gas_target"]
    metrics["n"] += batch_size
    metrics["gas_n"] += int(mask.sum())
    metrics["gas_correct"] += int((correct & mask).sum())
    if mask.any():
        encoded = batch["gas_target"][mask] * len(GAS_ORDER) + prediction[mask]
        metrics["confusion"] += torch.bincount(
            encoded, minlength=len(GAS_ORDER) ** 2,
        ).reshape(len(GAS_ORDER), len(GAS_ORDER)).cpu().numpy()
        probabilities = outputs["gas_logits"][mask].softmax(1)
        confidence, _ = probabilities.max(1)
        top2 = probabilities.topk(2, dim=1).values
        metrics["confidence"].extend(confidence.detach().cpu().tolist())
        metrics["confidence_correct"].extend(correct[mask].detach().cpu().tolist())
        metrics["margin"].extend((top2[:, 0] - top2[:, 1]).detach().cpu().tolist())
        metrics["nll"].extend(
            (-probabilities.gather(1, batch["gas_target"][mask, None]).log()).detach().cpu().ravel().tolist()
        )

    presence_counts = _binary_counts(
        torch.sigmoid(outputs["presence"]) >= 0.5, batch["presence_target"] >= 0.5,
    )
    recovery_mask = batch["recovery_mask"].bool()
    metrics["presence"].update(presence_counts)
    if recovery_mask.any():
        recovery_counts = _binary_counts(
            torch.sigmoid(outputs["recovery"])[recovery_mask] >= 0.5,
            batch["recovery_target"][recovery_mask] >= 0.5,
        )
        metrics["recovery"].update(recovery_counts)
    intensity_error = (outputs["intensity"] - batch["intensity_target"]).abs()
    metrics["intensity_abs"] += float(intensity_error.detach().sum())

    for i in range(batch_size):
        reason = batch["mask_reason"][i]
        metrics["mask_reason"][reason] += 1
        metrics["bucket"][batch["endpoint_bucket"][i]] += 1
        ppm = float(batch["local_total_ppm"][i])
        dominance = float(batch["dominant_fraction"][i])
        elapsed = float(batch["sniff_elapsed_s"][i])
        groups = [
            ("ppm", _bin_name(ppm, PPM_BINS)),
            ("dominance", _bin_name(dominance, DOMINANCE_BINS)),
            ("phase", batch["phase"][i]),
            ("exposure", batch["exposure_state"][i]),
            ("sniff_role", batch["sniff_role"][i]),
        ]
        if elapsed >= 0:
            groups.append(("sniff_elapsed", _bin_name(elapsed, TIME_BINS)))
        for dimension, value in groups:
            key = f"{dimension}:{value}"
            if bool(mask[i]):
                metrics["group_n"][key] += 1
                metrics["group_correct"][key] += int(bool(correct[i]))
        ppm_bin = _bin_name(ppm, PPM_BINS)
        metrics["intensity_n"][ppm_bin] += 1
        metrics["intensity_abs_bin"][ppm_bin] += float(intensity_error[i].detach())
        bucket = batch["endpoint_bucket"][i]
        if batch["sniff_role"][i] == "second" and bucket.startswith("second_sniff_"):
            offset = bucket.removeprefix("second_sniff_")
            if offset in SECOND_OFFSETS:
                metrics["second_offset_total"][offset] += 1
                if bool(mask[i]):
                    metrics["second_offset_n"][offset] += 1
                    metrics["second_offset_correct"][offset] += int(bool(correct[i]))
                    key = f"{batch['ordered_pair'][i]}|{offset}"
                    metrics["pair_offset_n"][key] += 1
                    metrics["pair_offset_correct"][key] += int(bool(correct[i]))

    metrics["loss_batches"] += 1
    for name, value in losses.items():
        metrics[f"loss_{name}"] += float(value.detach())


def _ece(confidence, correctness, bins=10):
    if not confidence:
        return 0.0
    confidence = np.asarray(confidence)
    correctness = np.asarray(correctness, float)
    result = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins
        selected = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= 1)
        if selected.any():
            result += selected.mean() * abs(correctness[selected].mean() - confidence[selected].mean())
    return float(result)


def _wilson_interval(correct: int, count: int, z: float = 1.959963984540054) -> list[float] | None:
    """Two-sided Wilson score interval for a binomial accuracy."""
    if count <= 0:
        return None
    proportion = correct / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
    ) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _grouped(metrics, prefix):
    output = {}
    for key, count in sorted(metrics["group_n"].items()):
        dimension, value = key.split(":", 1)
        if dimension == prefix:
            output[value] = {
                "n": count,
                "accuracy": metrics["group_correct"][key] / max(count, 1),
            }
    return output


def _finalize(metrics):
    confusion = metrics["confusion"]
    counts = confusion.sum(1)
    per_gas = np.divide(
        confusion.diagonal(), counts, out=np.zeros(len(GAS_ORDER), float), where=counts > 0,
    )
    batches = max(metrics["loss_batches"], 1)
    gas_n = max(metrics["gas_n"], 1)
    confidence = metrics["confidence"]
    mask_total = max(sum(metrics["mask_reason"].values()), 1)
    pair_matrix = {}
    for key, count in sorted(metrics["pair_offset_n"].items()):
        pair, offset = key.split("|")
        correct = metrics["pair_offset_correct"][key]
        pair_matrix.setdefault(pair, {})[offset] = {
            "n": count, "accuracy": correct / count,
            "accuracy_ci95_wilson": _wilson_interval(correct, count),
        }
    second_by_offset = {}
    for offset in SECOND_OFFSETS:
        total = metrics["second_offset_total"][offset]
        supervised = metrics["second_offset_n"][offset]
        correct = metrics["second_offset_correct"][offset]
        second_by_offset[offset] = {
            "eligible_endpoint_n": total,
            "supervised_n": supervised,
            "masked_n": total - supervised,
            "accuracy": correct / supervised if supervised else None,
            "accuracy_ci95_wilson": _wilson_interval(correct, supervised),
        }
    return {
        "loss": {
            name: metrics[f"loss_{name}"] / batches
            for name in ("total", "gas", "presence", "intensity", "recovery")
        },
        "gas": {
            "supervised_n": metrics["gas_n"],
            "accuracy": metrics["gas_correct"] / gas_n,
            "macro_accuracy": float(per_gas[counts > 0].mean()) if np.any(counts) else 0.0,
            "confusion_matrix": confusion.tolist(),
            "per_gas": {
                gas: {"n": int(counts[i]), "accuracy": float(per_gas[i])}
                for i, gas in enumerate(GAS_ORDER)
            },
            "nll": float(np.mean(metrics["nll"])) if metrics["nll"] else 0.0,
            "mean_top1_probability": float(np.mean(confidence)) if confidence else 0.0,
            "mean_top1_top2_margin": float(np.mean(metrics["margin"])) if metrics["margin"] else 0.0,
            "ece": _ece(confidence, metrics["confidence_correct"]),
            "accuracy_by_sniff_elapsed": _grouped(metrics, "sniff_elapsed"),
            "accuracy_by_local_total_ppm": _grouped(metrics, "ppm"),
            "accuracy_by_dominance": _grouped(metrics, "dominance"),
            "accuracy_by_trajectory_phase": _grouped(metrics, "phase"),
            "accuracy_by_exposure_state": _grouped(metrics, "exposure"),
            "first_vs_second": _grouped(metrics, "sniff_role"),
            "second_gas_accuracy_by_offset": second_by_offset,
            "second_gas_by_ordered_pair_and_offset": pair_matrix,
        },
        "presence": _binary_metrics(dict(metrics["presence"])),
        "recovery": _binary_metrics(dict(metrics["recovery"])),
        "intensity": {
            "logppm_mae": metrics["intensity_abs"] / max(metrics["n"], 1),
            "approximate_multiplicative_factor": math.exp(
                metrics["intensity_abs"] / max(metrics["n"], 1)
            ),
            "by_concentration_bin": {
                name: {
                    "n": count,
                    "logppm_mae": metrics["intensity_abs_bin"][name] / count,
                }
                for name, count in sorted(metrics["intensity_n"].items())
            },
        },
        "mask_coverage": {
            "total_endpoints": sum(metrics["mask_reason"].values()),
            "gas_supervised_fraction": metrics["gas_n"] / mask_total,
            "gas_masked_fraction": 1.0 - metrics["gas_n"] / mask_total,
            "by_reason": dict(sorted(metrics["mask_reason"].items())),
        },
        "endpoint_bucket_counts": dict(sorted(metrics["bucket"].items())),
    }


def _loss_kwargs(args):
    return {
        "lambda_gas": args.lambda_gas,
        "lambda_presence": args.lambda_presence,
        "lambda_intensity": args.lambda_intensity,
        "lambda_recovery": args.lambda_recovery,
    }


@torch.no_grad()
def evaluate_root(model, root, split, args, device, prefixes=None, phase=None):
    model.eval()
    output = {}
    for prefix in prefixes or PREFIXES[args.model]:
        dataset = PhysicalWindows(
            root, split, model_type=args.model, rate=args.rate,
            samples=args.eval_samples, seed=args.seed + 800_000,
            fixed_prefix_s=prefix, random_sampling=False,
            phase=phase,
            presence_threshold_ppm=args.presence_threshold_ppm,
            presence_negative_sample_fraction=0.0,
            gas_supervised_sample_fraction=0.0,
        )
        totals = _new_metrics()
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
            batch = _to_device(batch, device)
            predictions = _forward(model, batch, args.model)
            losses = _loss(predictions, batch, **_loss_kwargs(args))
            _accumulate(totals, predictions, batch, losses)
        output[f"{float(prefix):g}s"] = _finalize(totals)
    return output


def _build_architecture(args, baseline):
    return {
        "name": "PhysicalScentFormer4",
        "model_type": args.model,
        "gas_order": list(GAS_ORDER),
        "sensor_order": list(SENSOR_ORDER),
        "training_outputs": [
            "gas_logits_6", "gas_presence_logit", "local_logppm",
            "recovery_carryover_logit",
        ],
        "deployment_output": "embedding_only",
        "training_only_heads": [
            "gas_logits_6", "gas_presence_logit", "local_logppm",
            "recovery_carryover_logit",
        ],
        "normalization": "(voltage - clean_air_baseline) / clean_air_baseline",
        "clean_air_baseline_v": baseline.tolist(),
        "loss_weights": _loss_kwargs(args),
        "prefixes_s": list(PREFIXES[args.model]),
        "checkpoint_selection_prefix_s": args.selection_prefix_s,
        "phase_replay_mix": PHASE_MIX,
        "causal_history_never_reset_between_events": True,
        "meaningful_presence_threshold_ppm": args.presence_threshold_ppm,
        "training_endpoint_sampling": {
            "presence_negative_fraction": args.presence_negative_sample_fraction,
            "gas_supervised_class_balanced_fraction": args.gas_supervised_sample_fraction,
            "event_bucket_fraction": 1.0 - args.presence_negative_sample_fraction
            - args.gas_supervised_sample_fraction,
        },
        "recovery_loss_applicability": "masked for Phase A; enabled for B/C/D",
    }


def _forever(loader):
    while True:
        yield from loader


def _phase_weights(stage, args):
    if stage != "D":
        return PHASE_MIX[stage]
    values = np.asarray(args.final_mix, float)
    if len(values) != 4 or np.any(values <= 0):
        raise ValueError("--final-mix needs four positive A B C D weights")
    values /= values.sum()
    return dict(zip(("A", "B", "C", "D"), values.tolist()))


def _loaders(datasets, weights, batch_size):
    names = list(weights)
    sizes = {}
    remaining = batch_size
    for i, name in enumerate(names):
        size = remaining if i == len(names) - 1 else max(1, round(batch_size * weights[name]))
        sizes[name] = size
        remaining -= size
    if min(sizes.values()) < 1:
        raise ValueError("batch size too small for phase replay mix")
    return {
        name: _forever(DataLoader(datasets[name], batch_size=sizes[name], shuffle=False))
        for name in names
    }


def _merge(chunks, device):
    output = {}
    for key in chunks[0]:
        if isinstance(chunks[0][key], torch.Tensor):
            output[key] = torch.cat([chunk[key] for chunk in chunks]).to(device)
        else:
            output[key] = sum((list(chunk[key]) for chunk in chunks), [])
    return output


def _save(path, model, stage, epoch, args, architecture, metrics):
    torch.save({
        "model_state": model.state_dict(), "stage": stage, "epoch": epoch,
        "gas_order": GAS_ORDER, "sensor_order": SENSOR_ORDER,
        "config": vars(args), "architecture": architecture, "metrics": metrics,
    }, path)


def train(args):
    device = torch.device(args.device)
    roots = {
        "A": Path(args.phase_a), "B": Path(args.phase_b),
        "C": Path(args.phase_c), "D": Path(args.phase_d),
    }
    stages = tuple(args.training_phases.upper())
    required = sorted(set().union(*(_phase_weights(stage, args) for stage in stages)))
    datasets = {
        phase: PhysicalWindows(
            roots[phase], "train", model_type=args.model, rate=args.rate,
            samples=args.steps_per_epoch * args.batch_size,
            seed=args.seed + i * 100_000,
            phase=phase,
            presence_threshold_ppm=args.presence_threshold_ppm,
            presence_negative_sample_fraction=args.presence_negative_sample_fraction,
            gas_supervised_sample_fraction=args.gas_supervised_sample_fraction,
        )
        for i, phase in enumerate(required)
    }
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    baseline = next(iter(datasets.values())).baseline
    architecture = _build_architecture(args, baseline)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "architecture.json").write_text(json.dumps(architecture, indent=2) + "\n")
    history = {"schema_version": 4, "config": vars(args), "epochs": []}
    epochs = {"A": args.epochs_a, "B": args.epochs_b, "C": args.epochs_c, "D": args.epochs_d}

    for stage in stages:
        weights = _phase_weights(stage, args)
        loaders = _loaders(datasets, weights, args.batch_size)
        best_loss, best_score = float("inf"), -float("inf")
        for epoch in range(epochs[stage]):
            model.train()
            totals = _new_metrics()
            for _ in range(args.steps_per_epoch):
                batch = _merge([next(loaders[name]) for name in weights], device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda",
                ):
                    predictions = _forward(model, batch, args.model)
                    losses = _loss(predictions, batch, **_loss_kwargs(args))
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                _accumulate(totals, predictions, batch, losses)
            training = _finalize(totals)
            validation = evaluate_root(
                model, roots[stage], "validation", args, device,
                prefixes=(args.selection_prefix_s,), phase=stage,
            )[f"{args.selection_prefix_s:g}s"]
            score_terms = [(0.50, validation["gas"]["macro_accuracy"])]
            second = validation["gas"]["first_vs_second"].get("second")
            if second and second["n"]:
                score_terms.append((0.20, second["accuracy"]))
            if validation["presence"]["applicable"]:
                score_terms.append((0.15, validation["presence"]["balanced_accuracy"]))
            if validation["recovery"]["applicable"]:
                score_terms.append((0.15, validation["recovery"]["balanced_accuracy"]))
            score = sum(weight * value for weight, value in score_terms) / sum(
                weight for weight, _ in score_terms
            )
            record = {
                "stage": stage, "epoch": epoch + 1,
                "training": training, "validation": validation,
                "selection_score": score,
            }
            history["epochs"].append(record)
            (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            if validation["loss"]["total"] < best_loss:
                best_loss = validation["loss"]["total"]
                _save(out / f"phase_{stage.lower()}_best_loss.pt", model, stage, epoch + 1, args, architecture, validation)
            if score > best_score:
                best_score = score
                _save(out / f"phase_{stage.lower()}_best_score.pt", model, stage, epoch + 1, args, architecture, validation)
            print(
                f"[train] stage={stage} epoch={epoch + 1} "
                f"loss={training['loss']['total']:.4f} "
                f"gas={training['gas']['accuracy']:.4f} "
                f"presence_bal={training['presence']['balanced_accuracy']:.4f} "
                f"recovery_bal={training['recovery']['balanced_accuracy']:.4f}"
            )
        _save(out / f"phase_{stage.lower()}.pt", model, stage, epochs[stage], args, architecture, validation)
    (out / "metrics.json").write_text(json.dumps({
        stage: evaluate_root(model, roots[stage], "validation", args, device, phase=stage)
        for stage in required
    }, indent=2) + "\n")


def evaluate_checkpoint(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved = checkpoint["config"]
    for key in (
        "model", "rate", "dim", "heads", "layers", "dropout",
        "lambda_gas", "lambda_presence", "lambda_intensity", "lambda_recovery",
        "presence_threshold_ppm", "presence_negative_sample_fraction",
        "gas_supervised_sample_fraction",
    ):
        if key in saved:
            setattr(args, key, saved[key])
    model = build_model(args).to(device)
    model.load_state_dict(checkpoint["model_state"])
    roots = {
        "A": Path(args.phase_a), "B": Path(args.phase_b),
        "C": Path(args.phase_c), "D": Path(args.phase_d),
    }
    results = {
        "checkpoint": str(args.checkpoint), "split": args.evaluate,
        "phases": {
            phase: evaluate_root(model, root, args.evaluate, args, device, phase=phase)
            for phase, root in roots.items() if (root / args.evaluate).is_dir()
        },
    }
    destination = Path(args.eval_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[evaluate] wrote {destination}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("baseline", "dual_timescale"), default="baseline")
    parser.add_argument("--phase-a", default="datasets/odor_encoder/physical_phase_a_pilot_v2")
    parser.add_argument("--phase-b", default="datasets/odor_encoder/physical_phase_b_pilot_v1")
    parser.add_argument("--phase-c", default="datasets/odor_encoder/physical_phase_c_pilot_v1")
    parser.add_argument("--phase-d", default="datasets/odor_encoder/physical_phase_d_pilot_v1")
    parser.add_argument("--out-dir", default="checkpoints/physical_scentformer_v1")
    parser.add_argument("--training-phases", default="ABCD")
    parser.add_argument("--epochs-a", type=int, default=10)
    parser.add_argument("--epochs-b", type=int, default=15)
    parser.add_argument("--epochs-c", type=int, default=15)
    parser.add_argument("--epochs-d", type=int, default=20)
    parser.add_argument("--final-mix", type=float, nargs=4, default=(0.20, 0.25, 0.20, 0.35))
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--eval-samples", type=int, default=0, help="0 evaluates every stored endpoint")
    parser.add_argument(
        "--selection-prefix-s", type=float, default=5.0,
        help="causal history used for per-epoch validation and checkpoint selection",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-gas", type=float, default=1.0)
    parser.add_argument("--lambda-presence", type=float, default=0.3)
    parser.add_argument("--lambda-intensity", type=float, default=0.15)
    parser.add_argument("--lambda-recovery", type=float, default=0.3)
    parser.add_argument(
        "--presence-threshold-ppm", type=float, default=5.0,
        help="minimum local total ppm treated as meaningful VOC presence",
    )
    parser.add_argument(
        "--presence-negative-sample-fraction", type=float, default=0.20,
        help="training draws reserved for meaningful-presence negatives",
    )
    parser.add_argument(
        "--gas-supervised-sample-fraction", type=float, default=0.35,
        help="training draws reserved for class-balanced trusted gas endpoints",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", choices=("validation", "test"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-output", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.evaluate:
        if args.checkpoint is None or args.eval_output is None:
            raise SystemExit("--evaluate requires --checkpoint and --eval-output")
        evaluate_checkpoint(args)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
