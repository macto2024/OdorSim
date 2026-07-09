"""Task registry + scenario resolver for :func:`odor_sim.make`.

Maps a task string (e.g. ``"OdorLift"``) to its env class plus sensible
defaults, and a logical scenario name (e.g. ``"10x6_uniform"``) to the GADEN
environment configuration directory. This is the single place that knows the
string -> class / string -> path bindings, so ``make()`` stays thin.

No robosuite or ROS import happens at module import time; the env classes are
referenced lazily so this module (and thus ``list_tasks``) is importable
without MuJoCo on the path.
"""

from __future__ import annotations

from pathlib import Path

# Repo root: .../odor_sim/envs/registry.py -> parents[2] == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCENARIOS_DIR = _REPO_ROOT / "scenarios"


def _odor_lift():
    from odor_sim.envs.odor_lift import OdorLift

    return OdorLift


# string -> {factory, defaults}. ``cls`` is a zero-arg callable returning the
# env class (lazy import); ``defaults`` are merged under explicit make() kwargs.
# Objects carry their own dedicated recipes, so there is no default_recipe.
REGISTERED_TASKS: dict[str, dict] = {
    "OdorLift": {
        "cls": _odor_lift,
        "default_instruction": None,  # env generates one from the object name
    },
}


def list_tasks() -> list[str]:
    """Return the registered task names."""
    return sorted(REGISTERED_TASKS)


def get_task_spec(env: str) -> dict:
    """Return the registry entry for a task name (raises if unknown)."""
    if env not in REGISTERED_TASKS:
        raise KeyError(f"Unknown task {env!r}. Registered: {list_tasks()}")
    return REGISTERED_TASKS[env]


def resolve_scenario(scenario: str, config: str = "config1") -> Path:
    """Resolve a logical scenario name (or explicit path) to a config dir.

    Accepts either a logical name under ``scenarios/`` (e.g. ``"10x6_uniform"``,
    resolved to ``scenarios/<name>/environment_configurations/<config>``) or a
    direct path to an environment configuration directory (one containing
    ``config.yaml``). Returns an absolute :class:`~pathlib.Path`.
    """
    p = Path(scenario)
    if (p / "config.yaml").is_file():
        return p.resolve()

    config_dir = _SCENARIOS_DIR / scenario / "environment_configurations" / config
    if (config_dir / "config.yaml").is_file():
        return config_dir.resolve()

    raise FileNotFoundError(
        f"Could not resolve scenario {scenario!r}: no config.yaml at {config_dir} "
        f"or {p / 'config.yaml'}."
    )


def make_env(
    env: str,
    *,
    scenario_config_dir: "str | Path | None" = None,
    instruction: str | None = None,
    **env_kwargs,
):
    """Construct a registered task env, applying registry defaults.

    Args:
        env: registered task name (see :func:`list_tasks`).
        scenario_config_dir: GADEN config dir passed through to the env (drives
            its :class:`~odor_sim.config.frame_map.FrameMap`).
        instruction: task language label; falls back to the task default (which
            may be ``None``, letting the env generate one).
        **env_kwargs: forwarded verbatim to the env constructor (e.g.
            ``objects``).

    Returns:
        A constructed :class:`~odor_sim.envs.base.OdorManipulationEnv` subclass.
    """
    spec = get_task_spec(env)
    cls = spec["cls"]()

    instruction = instruction if instruction is not None else spec.get("default_instruction")

    kwargs = dict(env_kwargs)
    if instruction is not None:
        kwargs.setdefault("instruction", instruction)
    if scenario_config_dir is not None:
        kwargs.setdefault("scenario_config_dir", str(scenario_config_dir))

    return cls(**kwargs)
