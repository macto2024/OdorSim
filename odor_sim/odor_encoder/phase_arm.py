"""Phase C: scripted MuJoCo end-effector odor collection over three cups."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from odor_sim.bridge.collect_odor import _at_target, _eef_pos, _gripper_dof, _pose_delta_action
from odor_sim.odor_encoder.dataset import (
    GAS_ORDER, SENSOR_ORDER, MixtureSpec, assign_split, sample_mixture, write_episode,
)

_REPO = Path(__file__).resolve().parents[2]


def _recipe(spec: MixtureSpec) -> list[dict]:
    return [
        {"gas_type": gas, "strength": float(0.0 if ppm <= 0 else 0.05 + 0.95 * ppm / 100.0)}
        for gas, ppm in zip(GAS_ORDER, spec.concentration_ppm)
    ]


def _write_episode_catalog(root: Path, session_id: str, specs: list[MixtureSpec]) -> tuple[Path, Path, list[str]]:
    base = yaml.safe_load((_REPO / "odor_sim/config/liquids.yaml").read_text())
    names = [f"{session_id}_cup{i}" for i in range(len(specs))]
    recipes = {f"recipe_{name}": _recipe(spec) for name, spec in zip(names, specs)}
    base["liquids"] = {name: {"recipe": f"recipe_{name}"} for name in names}
    cfg = root / "generated_configs"
    cfg.mkdir(parents=True, exist_ok=True)
    liquid_path, recipe_path = cfg / f"{session_id}_liquids.yaml", cfg / f"{session_id}_recipes.yaml"
    liquid_path.write_text(yaml.safe_dump(base, sort_keys=False))
    recipe_path.write_text(yaml.safe_dump({"recipes": recipes}, sort_keys=False))
    return liquid_path, recipe_path, names


class ArmEpisode:
    def __init__(self, cosim, specs: list[MixtureSpec], names: list[str], rate: float, rng, gui: bool):
        self.cosim, self.env = cosim, cosim.env
        self.specs = dict(zip(names, specs))
        self.rate, self.rng, self.gui = rate, rng, gui
        self.obs = cosim.reset()
        self.rows = {k: [] for k in ("voltage", "rs_r0", "ppm", "time", "position", "phase", "presence", "fractions", "intensity", "clean", "valid")}

    def _zero(self) -> np.ndarray:
        action = np.zeros(int(self.env.action_dim), float)
        dof = _gripper_dof(self.env)
        if dof:
            action[-dof:] = -1.0
        return action

    def step(self, action: np.ndarray, phase: str, cup: str | None, valid: bool = True):
        self.obs, _, _, info = self.cosim.step(action)
        if self.gui:
            self.env.render()
        ppm_dict = info.get("ppm") or {}
        ppm = np.asarray([float(ppm_dict.get(g, 0.0)) for g in GAS_ORDER], np.float32)
        volts = info.get("enose_voltages") or {}
        self.rows["voltage"].append([float(volts.get(m, 0.0)) for m in SENSOR_ORDER])
        self.rows["rs_r0"].append([float(self.cosim._enoses[m].sensor.rs_r0) for m in SENSOR_ORDER])
        self.rows["ppm"].append(ppm)
        self.rows["time"].append(float(info.get("sim_time", len(self.rows["time"]) / self.rate)))
        self.rows["position"].append(np.asarray(self.env.get_enose_site_pose()[0], float))
        self.rows["phase"].append(phase)
        spec = self.specs.get(cup, MixtureSpec(np.zeros(6, np.float32), "pure"))
        self.rows["presence"].append(spec.presence)
        self.rows["fractions"].append(spec.fractions)
        self.rows["intensity"].append(np.log1p(float(ppm.sum())))
        self.rows["clean"].append(float(ppm.sum() < 1e-4))
        self.rows["valid"].append(bool(valid and cup is not None))

    def hold(self, seconds: float, phase: str, cup: str | None, target: np.ndarray | None = None):
        for _ in range(max(1, int(round(seconds * self.rate)))):
            action = self._zero() if target is None else _pose_delta_action(self.env, self.obs, target)
            self.step(action, phase, cup)

    def go(self, target: np.ndarray, phase: str, cup: str, max_steps: int = 180):
        for _ in range(max_steps):
            if _at_target(self.env, self.obs, target, xy_tol=0.025, z_tol=0.025):
                return
            self.step(_pose_delta_action(self.env, self.obs, target), phase, cup)
        raise RuntimeError(f"arm did not reach {phase} target")

    def visit(self, cup: str, *, dwell_s: float):
        source = np.asarray(self.env.get_object_world_poses()[cup][0], float)
        high = source + [0, 0, 0.30]
        self.go(high, "approach", cup)
        for height in self.rng.permutation([0.22, 0.15, 0.10]):
            hover = source + [0, 0, height]
            self.go(hover, "hover_height", cup)
            self.hold(0.4, "hover_height", cup, hover)
        center = source + [0, 0, 0.11]
        for offset in (-0.08, 0.08, 0.0):
            sweep = center + [0, offset, 0]
            self.go(sweep, "cross_cup_sweep", cup)
        self.hold(dwell_s, "dwell", cup, center)
        self.go(high, "retreat", cup)

    def arrays(self) -> tuple[dict, dict]:
        arrays = {
            "voltage": np.asarray(self.rows["voltage"], np.float32),
            "rs_r0": np.asarray(self.rows["rs_r0"], np.float32),
            "ppm": np.asarray(self.rows["ppm"], np.float32),
            "time_s": np.asarray(self.rows["time"], np.float32),
            "position": np.asarray(self.rows["position"], np.float32),
            "phase": np.asarray(self.rows["phase"]),
        }
        labels = {
            "label_presence": np.asarray(self.rows["presence"], np.float32),
            "label_fractions": np.asarray(self.rows["fractions"], np.float32),
            "label_intensity": np.asarray(self.rows["intensity"], np.float32),
            "label_clean": np.asarray(self.rows["clean"], np.float32),
            "label_valid": np.asarray(self.rows["valid"], bool),
        }
        return arrays, labels


def collect(args: argparse.Namespace) -> None:
    import odor_sim

    out = Path(args.out_dir)
    stop_index = args.start_index + args.episodes
    for index in range(args.start_index, stop_index):
        session_id = f"arm_{index:06d}"
        split = assign_split(session_id, args.seed)
        if (out / split / f"{session_id}.npz").is_file() and not args.overwrite:
            print(f"[phase-c] skip existing {session_id} (use --overwrite to rerun)")
            continue
        rng = np.random.default_rng(args.seed + index)
        specs = [sample_mixture(rng, split=split), *(sample_mixture(rng, split=split, force_nonempty=True) for _ in range(2))]
        liquid_path, recipe_path, names = _write_episode_catalog(out, session_id, specs)
        target = names[0]
        with odor_sim.make(
            "ClassifyLiquid", liquids=names, target_liquid=target,
            liquid_catalog_path=liquid_path, recipe_path=recipe_path,
            odor_mode="continuous", mox_model=list(SENSOR_ORDER), control_freq=args.rate,
            has_renderer=args.gui, has_offscreen_renderer=False, use_camera_obs=False,
            ignore_done=True, horizon=100000, seed=args.seed + index,
            scene_id=session_id, server_log_dir=args.server_log_dir,
        ) as cosim:
            episode = ArmEpisode(cosim, specs, names, args.rate, rng, args.gui)
            episode.hold(1.0, "settle", None)
            wrong_first = rng.random() < args.wrong_cup_probability
            order = [names[int(rng.integers(1, 3))], target] if wrong_first else [target]
            for cup in order:
                episode.visit(cup, dwell_s=float(rng.uniform(3.0, 12.0)))
            episode.hold(1.0, "final_retreat", target)
            arrays, labels = episode.arrays()
        write_episode(
            out, session_id=session_id, split=split, domain="mujoco_arm", rate=args.rate,
            labels=labels, metadata={
                "seed": args.seed + index, "cup_names": names, "target_cup": target,
                "wrong_cup_first": wrong_first,
                "source_concentrations_ppm": [s.concentration_ppm.tolist() for s in specs],
                "evaluation_group": specs[0].evaluation_group,
            }, **arrays,
        )
        print(f"[phase-c] {index + 1}/{args.episodes} wrong_first={wrong_first}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=300, help="Pilot target: 300")
    p.add_argument("--start-index", type=int, default=0, help="First episode id to generate")
    p.add_argument("--overwrite", action="store_true", help="Rerun and replace existing episode ids")
    p.add_argument("--out-dir", default="datasets/odor_encoder/arm")
    p.add_argument("--rate", type=float, default=20.0)
    p.add_argument("--wrong-cup-probability", type=float, default=0.65)
    p.add_argument("--server-log-dir", default=".roslog/odor_encoder_phase_c")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--seed", type=int, default=20000)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1 or args.start_index < 0 or args.rate <= 0 or not 0 <= args.wrong_cup_probability <= 1:
        raise SystemExit("invalid episode/rate/probability arguments")
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
