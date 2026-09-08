"""Inspect the held-out ethanol-to-acetone Phase-D acceptance trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

from odor_sim.odor_encoder.dataset import GAS_ORDER, SENSOR_ORDER
from odor_sim.odor_encoder.train_abcd import PhysicalWindows, _forward, build_model


def _find(root: Path, split: str) -> Path:
    for sidecar in sorted((root / split).glob("*.json")):
        metadata = json.loads(sidecar.read_text())
        if metadata.get("gas_a") == "ethanol" and metadata.get("gas_b") == "acetone":
            return sidecar.with_suffix(".npz")
    raise FileNotFoundError(f"no held-out ethanol->acetone trajectory in {root / split}")


def _load_model(checkpoint_path: Path | None, device: torch.device):
    if checkpoint_path is None:
        return None, None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = checkpoint["config"]
    args = SimpleNamespace(**saved)
    model = build_model(args).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, args


def _predict(model, model_args, signal: np.ndarray, endpoint: int, device):
    dataset = PhysicalWindows.__new__(PhysicalWindows)
    dataset.baseline_steps = int(round(20 * model_args.rate))
    dataset.recent_steps = int(round(10 * model_args.rate))
    dataset.history_steps = int(round(40 * model_args.rate))
    dataset.downsample = int(round(model_args.rate / 5.0))
    history = signal[:endpoint + 1]
    if model_args.model == "baseline":
        x, mask = dataset._pad(history, dataset.baseline_steps)
        batch = {
            "x": torch.from_numpy(x[None]).to(device),
            "padding_mask": torch.from_numpy(mask[None]).to(device),
        }
    else:
        recent, recent_mask, long, long_mask = dataset._dual(history)
        batch = {
            "recent": torch.from_numpy(recent[None]).to(device),
            "recent_mask": torch.from_numpy(recent_mask[None]).to(device),
            "history": torch.from_numpy(long[None]).to(device),
            "history_mask": torch.from_numpy(long_mask[None]).to(device),
        }
    with torch.no_grad():
        output = _forward(model, batch, model_args.model)
    probabilities = output["gas_logits"].softmax(1)[0].cpu().numpy()
    return {
        "predicted_gas": GAS_ORDER[int(probabilities.argmax())],
        "gas_probabilities": dict(zip(GAS_ORDER, probabilities.astype(float).tolist())),
        "presence_probability": float(torch.sigmoid(output["presence"])[0]),
        "predicted_local_total_ppm": float(torch.expm1(output["intensity"])[0]),
        "recovery_probability": float(torch.sigmoid(output["recovery"])[0]),
    }


def run(args) -> int:
    path = _find(args.phase_d, args.split)
    metadata = json.loads(path.with_suffix(".json").read_text())
    with np.load(path) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    device = torch.device(args.device)
    model, model_args = _load_model(args.checkpoint, device)
    endpoint_indices = np.flatnonzero(data["endpoint_eligible"])
    rows = []
    for endpoint in endpoint_indices:
        bucket = str(data["endpoint_bucket"][endpoint])
        selected = (
            bucket.startswith("first_sniff_")
            or bucket.startswith("second_sniff_")
            or bucket.startswith("departure1_")
        )
        if not selected:
            continue
        row = {
            "time_s": float(data["time_s"][endpoint]),
            "bucket": bucket,
            "phase": str(data["phase"][endpoint]),
            "local_total_ppm": float(data["local_total_ppm"][endpoint]),
            "dominant_gas": GAS_ORDER[int(data["dominant_gas_index"][endpoint])],
            "dominant_fraction": float(data["dominant_fraction"][endpoint]),
            "gas_loss_mask": bool(data["gas_loss_mask"][endpoint]),
            "mask_reason": str(data["gas_mask_reason"][endpoint]),
            "presence_target": int(data["label_presence"][endpoint]),
            "recovery_target": int(data["label_recovery"][endpoint]),
        }
        if model is not None:
            row["prediction"] = _predict(
                model, model_args, data["processed_mox_8ch"], int(endpoint), device,
            )
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"acceptance_{path.stem}"
    report = {
        "recording": str(path), "metadata": metadata,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "causal_endpoint_evaluation": rows,
    }
    (args.output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")

    time = data["time_s"]
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for gas_index, gas in enumerate(GAS_ORDER):
        axes[0].plot(time, data["ppm"][:, gas_index], label=gas, linewidth=1.0)
    axes[0].set_ylabel("Local ppm")
    axes[0].legend(ncol=3, fontsize=8)
    for channel, sensor in enumerate(SENSOR_ORDER):
        axes[1].plot(time, data["processed_mox_8ch"][:, channel], label=sensor, linewidth=0.8)
    axes[1].set_ylabel("Processed MOX")
    axes[1].legend(ncol=4, fontsize=7)
    axes[2].step(time, data["label_presence"], where="post", label="presence")
    axes[2].step(time, data["label_recovery"], where="post", label="recovery")
    axes[2].step(time, data["gas_loss_mask"], where="post", label="gas loss mask")
    axes[2].scatter(
        time[endpoint_indices], np.full(len(endpoint_indices), 1.08), s=8,
        color="black", label="sampled endpoint",
    )
    axes[2].set_ylim(-0.1, 1.2)
    axes[2].set_ylabel("Target / mask")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(ncol=4, fontsize=8)
    figure.suptitle("Held-out ethanol -> acetone physical carryover trajectory")
    figure.tight_layout()
    figure.savefig(args.output_dir / f"{stem}.pdf")
    plt.close(figure)
    print(f"recording: {path}")
    print(f"report: {args.output_dir / f'{stem}.json'}")
    print(f"plot: {args.output_dir / f'{stem}.pdf'}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-d", type=Path, default=Path("datasets/odor_encoder/physical_phase_d_pilot_v1"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/odor_encoder/physical_acceptance"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
