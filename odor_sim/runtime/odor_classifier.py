"""Live ScentFormer odor classification from MOX voltage streams.

Loads the ScentFormerAB checkpoint (``Smellnet_chkpt/phase_b_best_loss.pt``)
and runs inference on rolling 20 Hz eight-channel voltage windows with the
same preprocessing used during training in ``Smellnet_chkpt/train_ab.py``.

By default every inference step batches all ready fixed prefixes (3/5/7/10/15/20 s)
in one GPU forward pass.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from odor_sim.sensors.mox_pid import MOXSensor

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = _REPO_ROOT / "Smellnet_chkpt" / "phase_b_best_loss.pt"
PREFIX_SECONDS = (3, 5, 7, 10, 15, 20)


class OdorTransformer(nn.Module):
    """ScentFormerAB temporal encoder (vendored from Smellnet_chkpt/train_ab.py)."""

    def __init__(
        self,
        max_steps: int,
        n_channels: int,
        dim: int = 256,
        heads: int = 8,
        layers: int = 4,
        dropout: float = 0.1,
        n_gases: int = 6,
    ):
        super().__init__()
        self.input = nn.Linear(n_channels, dim)
        self.scent_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.position = nn.Parameter(torch.randn(1, max_steps + 1, dim) * 0.01)
        block = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(dim)
        self.presence = nn.Linear(dim, n_gases)
        self.fractions = nn.Linear(dim, n_gases)
        self.intensity = nn.Linear(dim, 1)
        self.clean = nn.Linear(dim, 1)
        self.recovery = nn.Linear(dim, 1)

    def forward(self, x, padding_mask):
        temporal = self.input(x)
        scent = self.scent_token.expand(x.shape[0], -1, -1)
        z = torch.cat((temporal, scent), dim=1)
        z = z + self.position[:, : z.shape[1]]
        scent_is_valid = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device)
        padding_mask = torch.cat((padding_mask, scent_is_valid), dim=1)
        z = self.encoder(z, src_key_padding_mask=padding_mask)
        z = self.norm(z[:, -1])
        return (
            self.presence(z),
            self.fractions(z),
            F.softplus(self.intensity(z).squeeze(-1)),
            self.clean(z).squeeze(-1),
            self.recovery(z).squeeze(-1),
        )


@dataclass(frozen=True)
class OdorPrediction:
    """One ScentFormer inference result for a single fixed prefix."""

    gas_names: tuple[str, ...]
    primary_confidence: np.ndarray
    presence: np.ndarray
    clean_air: float
    recovery: float
    intensity_ppm: float
    top_gas: str
    top_confidence: float
    window_s: float
    prefix_steps: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "gas_names": list(self.gas_names),
            "primary_confidence": self.primary_confidence.tolist(),
            "presence": self.presence.tolist(),
            "clean_air": float(self.clean_air),
            "recovery": float(self.recovery),
            "intensity_ppm": float(self.intensity_ppm),
            "top_gas": self.top_gas,
            "top_confidence": float(self.top_confidence),
            "window_s": float(self.window_s),
            "prefix_steps": int(self.prefix_steps),
        }


@dataclass(frozen=True)
class MultiWindowOdorPrediction:
    """Parallel fixed-prefix ScentFormer outputs (3/5/7/10/15/20 s)."""

    gas_names: tuple[str, ...]
    by_window_s: tuple[int, ...]
    predictions: tuple[OdorPrediction, ...]
    device: str

    @property
    def primary(self) -> OdorPrediction:
        """Prefer the 20 s window; else the longest ready prefix."""
        by_s = {int(round(p.window_s)): p for p in self.predictions}
        if 20 in by_s:
            return by_s[20]
        return self.predictions[-1]

    @property
    def top_gas(self) -> str:
        return self.primary.top_gas

    @property
    def top_confidence(self) -> float:
        return self.primary.top_confidence

    def prediction_at(self, window_s: int) -> OdorPrediction | None:
        for pred in self.predictions:
            if int(round(pred.window_s)) == window_s:
                return pred
        return None

    def as_dict(self) -> dict[str, Any]:
        primary = self.primary
        windows = {
            str(int(round(pred.window_s))): pred.as_dict() for pred in self.predictions
        }
        return {
            "mode": "multi_window",
            "device": self.device,
            "gas_names": list(self.gas_names),
            "windows": windows,
            "ready_window_s": [int(round(p.window_s)) for p in self.predictions],
            "top_gas": primary.top_gas,
            "top_confidence": float(primary.top_confidence),
            "window_s": float(primary.window_s),
            "clean_air": float(primary.clean_air),
        }


def resolve_checkpoint_path(path: str | Path | None) -> Path:
    """Return an existing checkpoint path (default: phase_b_best_loss.pt)."""
    if path is None or path is True:
        ckpt = DEFAULT_CHECKPOINT
    else:
        ckpt = Path(path)
        if not ckpt.is_absolute():
            ckpt = _REPO_ROOT / ckpt
    if not ckpt.is_file():
        raise FileNotFoundError(f"ScentFormer checkpoint not found: {ckpt}")
    return ckpt


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_scentformer_classifier(
    checkpoint_path: str | Path | None,
    *,
    mox_models: list[str],
    device: str | torch.device | None = None,
    infer_every_n: int = 10,
) -> "ScentFormerClassifier":
    """Construct a :class:`ScentFormerClassifier` from a checkpoint path."""
    return ScentFormerClassifier(
        resolve_checkpoint_path(checkpoint_path),
        mox_models=mox_models,
        device=device,
        infer_every_n=infer_every_n,
    )


class ScentFormerClassifier:
    """Rolling-window ScentFormer inference on live MOX voltages."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        mox_models: list[str],
        device: str | torch.device | None = None,
        infer_every_n: int = 10,
    ):
        ckpt_path = Path(checkpoint_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.checkpoint_path = ckpt_path
        self.config = dict(ckpt["config"])
        self.sensor_order: list[str] = list(ckpt["sensor_order"])
        self.gas_order: tuple[str, ...] = tuple(ckpt["gas_order"])
        live_models = list(mox_models)
        if live_models != self.sensor_order:
            raise ValueError(
                "MOX model order must match the ScentFormer checkpoint sensor_order "
                f"{self.sensor_order!r}; got {live_models!r}. "
                "Omit --mox-model to use the default eight-sensor OVLA array."
            )

        self.rate = float(self.config["rate"])
        self.max_history_s = float(self.config["max_history_s"])
        self.max_steps = int(round(self.max_history_s * self.rate))
        self.prefix_steps = [
            int(round(s * self.rate))
            for s in PREFIX_SECONDS
            if s <= self.max_history_s
        ]
        self.prefix_seconds = tuple(
            int(round(steps / self.rate)) for steps in self.prefix_steps
        )
        if not self.prefix_steps:
            raise ValueError(f"no valid prefix durations for max_history_s={self.max_history_s}")

        self.infer_every_n = max(1, int(infer_every_n))
        self.device = _resolve_device(device)
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        baseline = [
            MOXSensor(model=m, rate=self.rate).baseline_voltage() for m in self.sensor_order
        ]
        self.baseline = np.asarray(baseline, dtype=np.float32)

        self.model = OdorTransformer(
            self.max_steps,
            len(self.sensor_order),
            dim=int(self.config["dim"]),
            heads=int(self.config["heads"]),
            layers=int(self.config["layers"]),
            dropout=float(self.config.get("dropout", 0.1)),
            n_gases=len(self.gas_order),
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self._buffer: deque[np.ndarray] = deque(maxlen=self.max_steps)
        self._step_count = 0
        print(
            f"[classifier] ScentFormer on {self.device} "
            f"(batch prefixes {list(self.prefix_seconds)}s, infer every {self.infer_every_n} steps)"
        )

    @property
    def min_prefix_steps(self) -> int:
        return self.prefix_steps[0]

    def reset(self) -> None:
        """Clear the rolling voltage buffer."""
        self._buffer.clear()
        self._step_count = 0

    def _ready_prefixes(self, n_samples: int) -> list[int]:
        return [p for p in self.prefix_steps if self.min_prefix_steps <= p <= n_samples]

    def _build_window(self, prefix: int) -> tuple[np.ndarray, np.ndarray]:
        raw = np.stack(list(self._buffer)[-prefix:], axis=0).astype(np.float32)
        x = (raw - self.baseline) / np.maximum(self.baseline, 1e-4)
        padded = np.zeros((self.max_steps, len(self.sensor_order)), dtype=np.float32)
        mask = np.ones(self.max_steps, dtype=bool)
        padded[-prefix:] = x
        mask[-prefix:] = False
        return padded, mask

    def _build_batch(self, prefixes: list[int]) -> tuple[np.ndarray, np.ndarray]:
        batch_padded = np.zeros(
            (len(prefixes), self.max_steps, len(self.sensor_order)),
            dtype=np.float32,
        )
        batch_mask = np.ones((len(prefixes), self.max_steps), dtype=bool)
        for i, prefix in enumerate(prefixes):
            padded, mask = self._build_window(prefix)
            batch_padded[i] = padded
            batch_mask[i] = mask
        return batch_padded, batch_mask

    def _decode_batch(
        self,
        prefixes: list[int],
        presence_logits: torch.Tensor,
        fraction_logits: torch.Tensor,
        intensity: torch.Tensor,
        clean_logits: torch.Tensor,
        recovery_logits: torch.Tensor,
    ) -> tuple[OdorPrediction, ...]:
        primary = torch.softmax(fraction_logits, dim=-1).cpu().numpy()
        presence = torch.sigmoid(presence_logits).cpu().numpy()
        clean_air = torch.sigmoid(clean_logits).cpu().numpy()
        recovery = torch.sigmoid(recovery_logits).cpu().numpy()
        intensity_ppm = torch.expm1(intensity).cpu().numpy()
        out: list[OdorPrediction] = []
        for i, prefix in enumerate(prefixes):
            top_i = int(primary[i].argmax())
            out.append(
                OdorPrediction(
                    gas_names=self.gas_order,
                    primary_confidence=primary[i],
                    presence=presence[i],
                    clean_air=float(clean_air[i]),
                    recovery=float(recovery[i]),
                    intensity_ppm=float(intensity_ppm[i]),
                    top_gas=self.gas_order[top_i],
                    top_confidence=float(primary[i, top_i]),
                    window_s=prefix / self.rate,
                    prefix_steps=prefix,
                )
            )
        return tuple(out)

    @torch.no_grad()
    def _infer_batch(self, prefixes: list[int]) -> tuple[OdorPrediction, ...]:
        padded, mask = self._build_batch(prefixes)
        x = torch.from_numpy(padded).to(self.device)
        m = torch.from_numpy(mask).to(self.device)
        presence_logits, fraction_logits, intensity, clean_logits, recovery_logits = self.model(
            x, m
        )
        return self._decode_batch(
            prefixes,
            presence_logits,
            fraction_logits,
            intensity,
            clean_logits,
            recovery_logits,
        )

    def update(self, voltages: dict[str, float]) -> MultiWindowOdorPrediction | None:
        """Append one voltage sample; batch-infer all ready fixed prefixes."""
        vec = np.asarray(
            [float(voltages.get(name, 0.0)) for name in self.sensor_order],
            dtype=np.float32,
        )
        self._buffer.append(vec)
        self._step_count += 1
        if self._step_count % self.infer_every_n != 0:
            return None
        prefixes = self._ready_prefixes(len(self._buffer))
        if not prefixes:
            return None
        predictions = self._infer_batch(prefixes)
        return MultiWindowOdorPrediction(
            gas_names=self.gas_order,
            by_window_s=tuple(int(round(p.window_s)) for p in predictions),
            predictions=predictions,
            device=str(self.device),
        )
