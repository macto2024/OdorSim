"""Train SmellNet ScentFormer (Transformer) on OdorSim sniff windows.

Expects the collector output::

    datasets/odor_sniffs/<run>/windows/dataset.npz
    # X: (N, Tw, C), y: (N,), sniff_id: (N,), label_names: (K,)

Splits **by sniff id** (not random windows) to avoid train/val leakage.

Example::

    source setup/activate.sh
    python -m odor_sim.bridge.train_scentformer \\
      --data datasets/odor_sniffs/voc6_continuous/windows/dataset.npz \\
      --epochs 40 --batch-size 32 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_smellnet_models(cli_path: str | None) -> Path:
    if cli_path:
        return Path(cli_path)
    for candidate in (
        Path("/root/SmellNet/models"),
        _REPO_ROOT.parent / "SmellNet" / "models",
        _REPO_ROOT / "third_party" / "SmellNet" / "models",
    ):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "SmellNet models not found. Expected /root/SmellNet/models "
        "or pass --smellnet-models /path/to/SmellNet/models"
    )


def _import_smellnet(models_dir: Path):
    if not models_dir.is_dir():
        raise FileNotFoundError(f"SmellNet models not found at {models_dir}")
    sys.path.insert(0, str(models_dir))
    from models import Transformer  # noqa: WPS433
    from train import train  # noqa: WPS433

    return Transformer, train


def _sniff_split(
    sniff_id: np.ndarray,
    *,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (train, val) over windows, split by unique sniff."""
    rng = np.random.default_rng(seed)
    unique = np.unique(sniff_id)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_frac)))
    val_set = set(unique[:n_val].tolist())
    is_val = np.array([s in val_set for s in sniff_id], dtype=bool)
    return ~is_val, is_val


def _standardize_train(
    Xtr: np.ndarray, Xva: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Channel-wise StandardScaler fit on train windows only. X: (N,T,C)."""
    n, t, c = Xtr.shape
    flat = Xtr.reshape(n * t, c)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

    def apply(X):
        return ((X - mean) / std).astype(np.float32)

    return apply(Xtr), apply(Xva), stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ScentFormer on OdorSim sniff windows")
    p.add_argument(
        "--data",
        type=str,
        default="datasets/odor_sniffs/voc6_continuous/windows/dataset.npz",
        help="Path to windows/dataset.npz from collect_odor",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2, help="Fraction of sniffs for val")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Checkpoint / metrics dir (default: <data-parent>/scentformer_run)",
    )
    p.add_argument(
        "--smellnet-models",
        type=str,
        default=None,
        help="Path to SmellNet/models (default: /root/SmellNet/models)",
    )
    p.add_argument("--tf-dim", type=int, default=256)
    p.add_argument("--tf-heads", type=int, default=8)
    p.add_argument("--tf-layers", type=int, default=3)
    p.add_argument("--tf-dropout", type=float, default=0.1)
    p.add_argument(
        "--baseline-subtract",
        action="store_true",
        help="Subtract first timestep per window (SmellNet-style baseline).",
    )
    return p


@torch.no_grad()
def _eval_acc(model, loader, device) -> dict:
    model.eval()
    correct1 = correct5 = n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct1 += (pred == y).sum().item()
        k = min(5, logits.size(1))
        topk = torch.topk(logits, k=k, dim=1).indices
        correct5 += (topk == y.unsqueeze(1)).any(dim=1).sum().item()
        n += y.numel()
    return {
        "acc@1": correct1 / max(n, 1),
        "acc@5": correct5 / max(n, 1),
        "n": n,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = _REPO_ROOT / data_path
    if not data_path.is_file():
        raise SystemExit(f"Missing dataset: {data_path}")

    models_dir = _resolve_smellnet_models(args.smellnet_models)
    Transformer, train_fn = _import_smellnet(models_dir)

    raw = np.load(data_path, allow_pickle=True)
    X = np.asarray(raw["X"], dtype=np.float32)
    y = np.asarray(raw["y"], dtype=np.int64)
    sniff_id = np.asarray(raw["sniff_id"])
    label_names = [str(s) for s in np.asarray(raw["label_names"]).tolist()]
    n, tw, c = X.shape
    k = len(label_names)
    print(f"[train_scentformer] loaded {data_path}")
    print(f"  X={X.shape} classes={label_names}")

    if args.baseline_subtract:
        X = X - X[:, :1, :]

    train_mask, val_mask = _sniff_split(sniff_id, val_frac=args.val_frac, seed=args.seed)
    Xtr, Xva, scaler = _standardize_train(X[train_mask], X[val_mask])
    ytr, yva = y[train_mask], y[val_mask]
    print(
        f"  split by sniff: train_windows={Xtr.shape[0]} val_windows={Xva.shape[0]} "
        f"(val_frac={args.val_frac})"
    )

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = Transformer(
        input_dim=c,
        model_dim=args.tf_dim,
        num_classes=k,
        num_heads=args.tf_heads,
        num_layers=args.tf_layers,
        dropout=args.tf_dropout,
    )

    out_dir = Path(args.out_dir) if args.out_dir else data_path.parent / "scentformer_run"
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_scentformer] device={device} epochs={args.epochs} → {out_dir}")
    train_fn(
        model,
        train_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        dtype=torch.float32,
    )

    metrics = {
        "train": _eval_acc(model, train_loader, device),
        "val": _eval_acc(model, val_loader, device),
        "label_names": label_names,
        "X_shape": list(X.shape),
        "window_steps": tw,
        "channels": c,
        "baseline_subtract": bool(args.baseline_subtract),
        "data": str(data_path),
    }
    print(
        f"[train_scentformer] train acc@1={metrics['train']['acc@1']:.3f}  "
        f"val acc@1={metrics['val']['acc@1']:.3f}  "
        f"val acc@5={metrics['val']['acc@5']:.3f}"
    )

    ckpt = {
        "model_state": model.state_dict(),
        "label_names": label_names,
        "scaler": scaler,
        "config": {
            "input_dim": c,
            "model_dim": args.tf_dim,
            "num_classes": k,
            "num_heads": args.tf_heads,
            "num_layers": args.tf_layers,
            "dropout": args.tf_dropout,
            "window_steps": tw,
            "baseline_subtract": bool(args.baseline_subtract),
        },
    }
    torch.save(ckpt, out_dir / "scentformer.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[train_scentformer] wrote {out_dir / 'scentformer.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
