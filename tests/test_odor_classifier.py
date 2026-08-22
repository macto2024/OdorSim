"""Unit tests for live ScentFormer odor classification."""

from __future__ import annotations

import sys

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except ImportError:
    pytest.skip("torch required for ScentFormer classifier tests", allow_module_level=True)

from odor_sim.config.gas_types import EXCEL_VOC_GAS_NAMES
from odor_sim.runtime.classifier_monitor import ClassifierMonitor, parse_classifier_monitor_spec
from odor_sim.runtime.odor_classifier import (
    DEFAULT_CHECKPOINT,
    PREFIX_SECONDS,
    ScentFormerClassifier,
    load_scentformer_classifier,
    resolve_checkpoint_path,
)
from odor_sim.sensors.mox_pid import DEFAULT_MOX_MODELS

pytestmark = pytest.mark.skipif(
    not DEFAULT_CHECKPOINT.is_file(),
    reason=f"checkpoint missing: {DEFAULT_CHECKPOINT}",
)


def test_resolve_checkpoint_path_default():
    path = resolve_checkpoint_path(None)
    assert path == DEFAULT_CHECKPOINT
    assert path.is_file()


def test_parse_classifier_monitor_spec():
    assert parse_classifier_monitor_spec(False) == {"enabled": False}
    assert parse_classifier_monitor_spec(True) == {"enabled": True, "log": True, "plot": True}
    assert parse_classifier_monitor_spec("log") == {"enabled": True, "log": True, "plot": False}


@pytest.fixture(scope="module")
def classifier() -> ScentFormerClassifier:
    return load_scentformer_classifier(
        None,
        mox_models=list(DEFAULT_MOX_MODELS),
        infer_every_n=1,
        device="cpu",
    )


def _feed_voltages(clf: ScentFormerClassifier, n: int, voltage: float = 1.0):
    last = None
    volts = {m: voltage for m in clf.sensor_order}
    for _ in range(n):
        last = clf.update(volts)
    return last


def test_short_buffer_returns_none(classifier: ScentFormerClassifier):
    classifier.reset()
    pred = _feed_voltages(classifier, classifier.min_prefix_steps - 1)
    assert pred is None


def test_clean_air_high_clean_probability(classifier: ScentFormerClassifier):
    classifier.reset()
    pred = _feed_voltages(classifier, classifier.max_steps, voltage=1.0)
    assert pred is not None
    primary = pred.primary
    assert primary.clean_air > 0.5
    assert abs(float(primary.primary_confidence.sum()) - 1.0) < 1e-5


def test_primary_confidence_sums_to_one(classifier: ScentFormerClassifier):
    classifier.reset()
    volts = {m: 1.0 + 0.05 * (i + 1) for i, m in enumerate(classifier.sensor_order)}
    last = None
    for _ in range(classifier.max_steps):
        last = classifier.update(volts)
    assert last is not None
    for pred in last.predictions:
        assert abs(float(pred.primary_confidence.sum()) - 1.0) < 1e-5
        assert pred.top_gas in EXCEL_VOC_GAS_NAMES


def test_multi_window_batch_sizes(classifier: ScentFormerClassifier):
    classifier.reset()
    n = int(round(8 * classifier.rate))
    pred = _feed_voltages(classifier, n, voltage=1.02)
    assert pred is not None
    assert pred.by_window_s == (3, 5, 7)

    classifier.reset()
    full = _feed_voltages(classifier, classifier.max_steps, voltage=1.02)
    assert full is not None
    assert full.by_window_s == classifier.prefix_seconds


def test_multi_window_as_dict(classifier: ScentFormerClassifier):
    classifier.reset()
    pred = _feed_voltages(classifier, classifier.max_steps, voltage=1.0)
    assert pred is not None
    d = pred.as_dict()
    assert d["mode"] == "multi_window"
    assert set(d["windows"].keys()) == {str(s) for s in classifier.prefix_seconds}
    assert d["top_gas"] == pred.prediction_at(20).top_gas


def test_mox_order_mismatch_raises():
    with pytest.raises(ValueError, match="sensor_order"):
        load_scentformer_classifier(
            None, mox_models=["101", "TGS2600"], infer_every_n=1, device="cpu"
        )


def test_classifier_monitor_log(capsys):
    mon = ClassifierMonitor(list(EXCEL_VOC_GAS_NAMES), log=True, plot=False)
    mon.set_ground_truth({"left": "ethanol", "right": "toluene"})
    from odor_sim.runtime.odor_classifier import MultiWindowOdorPrediction, OdorPrediction

    def _pred(window_s: float, gas: str, conf: float) -> OdorPrediction:
        probs = np.zeros(len(EXCEL_VOC_GAS_NAMES), dtype=np.float32)
        probs[list(EXCEL_VOC_GAS_NAMES).index(gas)] = conf
        return OdorPrediction(
            gas_names=tuple(EXCEL_VOC_GAS_NAMES),
            primary_confidence=probs,
            presence=np.zeros(len(EXCEL_VOC_GAS_NAMES), dtype=np.float32),
            clean_air=0.1,
            recovery=0.0,
            intensity_ppm=1.0,
            top_gas=gas,
            top_confidence=conf,
            window_s=window_s,
            prefix_steps=int(window_s * 20),
        )

    batch = MultiWindowOdorPrediction(
        gas_names=tuple(EXCEL_VOC_GAS_NAMES),
        by_window_s=(3, 10),
        predictions=(_pred(3, "ethanol", 0.7), _pred(10, "ethanol", 0.9)),
        device="cuda:0",
    )
    mon.record(10.0, batch)
    out = capsys.readouterr().out
    assert "[classifier]" in out
    assert "device=cuda:0" in out
    assert "3s  ethanol" in out
    assert "GT left=ethanol" in out


def test_cup_ground_truth_helper():
    from odor_sim.bridge.teleop import cup_ground_truth

    class _FakeEnv:
        _place_target_name = "plate"

        def spawn_layout(self):
            return {
                "ethanol": {"pos": [0.16, -0.12, 0.85]},
                "toluene": {"pos": [0.16, 0.12, 0.85]},
                "plate": {"pos": [0.0, 0.0, 0.85]},
            }

    gt = cup_ground_truth(_FakeEnv())
    assert gt == {"left": "ethanol", "right": "toluene"}


def test_make_dry_run_classifier_smoke():
    """Headless cosim smoke: classifier runs on live voltages without GADEN."""
    import odor_sim as odorsim

    with odorsim.make(
        "ClassifyLiquidPlaceOVLA",
        auto_start_gaden=False,
        bridge=False,
        export=False,
        odor_mode="continuous",
        odor_classifier={"classifier_monitor": "log", "device": "cpu"},
        liquids=["ethanol", "toluene"],
        place_target="plate",
        has_renderer=False,
        control_freq=20,
    ) as cosim:
        cosim.reset()
        prediction = None
        for _ in range(400):
            _, _, _, info = cosim.step(cosim.zero_action())
            if "odor_prediction" in info:
                prediction = info["odor_prediction"]
        assert prediction is not None
        assert prediction["mode"] == "multi_window"
        assert prediction["top_gas"] in EXCEL_VOC_GAS_NAMES
        assert len(prediction["windows"]) == len(PREFIX_SECONDS)


def _run_all():
    return pytest.main([__file__, "-q"])


if __name__ == "__main__":
    sys.exit(_run_all())
