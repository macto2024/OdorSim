"""``odor_sim.make`` — one call brings up the full GADEN co-simulation.

Hides the three-terminal Phase 4 workflow (export scene -> ``ros2 run
odor_gaden_rt`` -> connect bridge) behind a single entry point that returns an
:class:`~odor_sim.runtime.session.OdorCosimSession`::

    import odor_sim as odorsim

    with odorsim.make("OdorLift", recipe="ripe_fruit") as cosim:
        obs = cosim.reset()
        obs, reward, done, info = cosim.step(action)
        print(info["ppm"])   # ground-truth per-gas ppm at the EE

Design rules:

  * The GADEN scene is exported from the **env's** ``scene_builder`` after the
    env is constructed, so the server's source *i* is the env's source *i*
    index-for-index — impossible to desync a parallel ``--recipe`` list.
  * The server runs as a managed subprocess (lockstep); ``make()`` must be
    called from a shell that sourced ``setup/activate.sh``.
  * Composition, not subclassing: the session wraps env + bridge + server.

Open decisions resolved for v1:
  * **Export target:** the scene YAML is written into the scenario config dir
    (default ``scene_id = f"{env}_{recipe}"``), so its relative CAD/wind paths
    keep resolving. These generated files are gitignored.
  * **Server reuse:** ``connect_only=True`` attaches the bridge to an
    already-running server and does not spawn or stop one.
  * **Observation contract:** ppm is exposed in ``info`` (not ``obs``); the
    final schema is fixed in Phase 5.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from odor_sim.envs.registry import get_task_spec, make_env, resolve_scenario
from odor_sim.runtime.odor_monitor import OdorMonitor, parse_odor_monitor_spec
from odor_sim.runtime.session import OdorCosimSession

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _gas_types_from_env(env) -> list[str]:
    """Unique GADEN gas names in source order (matches bridge / teleop)."""
    sb = getattr(env, "scene_builder", None)
    if sb is None:
        return []
    return list(dict.fromkeys(s.component.gaden_gas_name() for s in sb.sources))


def _build_odor_monitor(spec, gas_types: list[str]):
    """Construct an :class:`OdorMonitor` from a make() ``odor_monitor`` spec."""
    opts = parse_odor_monitor_spec(spec)
    if not opts.pop("enabled", False):
        return None
    if not gas_types:
        return None
    return OdorMonitor(gas_types, **opts)


def _default_scene_id(env: str, recipe: "str | None") -> str:
    raw = f"{env}_{recipe}" if recipe else env
    return re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_").lower()


def _ensure_writable_ros_log_dir() -> None:
    """Point ``ROS_LOG_DIR`` at a writable dir if the default is not.

    rclpy (used by the in-process bridge) and the server subprocess both try to
    open a log file under ``~/.ros/log`` at init. Under a sandbox / restricted
    ``$HOME`` that write fails. If ``ROS_LOG_DIR`` is already set we respect it;
    otherwise we only override when the default location is not writable, so
    normal environments keep ROS's usual logging behavior.
    """
    if os.environ.get("ROS_LOG_DIR"):
        return
    default = Path(os.environ.get("ROS_HOME", Path.home() / ".ros")) / "log"
    try:
        default.mkdir(parents=True, exist_ok=True)
        probe = default / ".odorsim_write_test"
        probe.touch()
        probe.unlink()
        return  # default is writable; leave rclpy on its normal path
    except OSError:
        fallback = _REPO_ROOT / ".roslog" / "rcl"
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ["ROS_LOG_DIR"] = str(fallback)


def make(
    env: str,
    *,
    recipe: "str | None" = None,
    scenario: str = "10x6_uniform",
    scenario_config: str = "config1",
    scene_id: "str | None" = None,
    auto_start_gaden: bool = True,
    connect_only: bool = False,
    bridge: bool = True,
    export: bool = True,
    server_log_dir: "str | Path | None" = None,
    wait_timeout: float = 60.0,
    step_on_timer: bool = False,
    publish_markers: bool = False,
    odor_monitor=False,
    **env_kwargs,
) -> OdorCosimSession:
    """Construct an odor co-simulation session.

    Args:
        env: registered task name (e.g. ``"OdorLift"``; see
            :func:`odor_sim.envs.registry.list_tasks`).
        recipe: VOC recipe name; defaults to the task's ``default_recipe``.
        scenario: logical scenario name under ``scenarios/`` or a direct path to
            a GADEN environment configuration directory.
        scenario_config: which ``environment_configurations/<name>`` to use when
            ``scenario`` is a logical name.
        scene_id: scene file name to export/load; defaults to
            ``f"{env}_{recipe}"`` (lowercased).
        auto_start_gaden: spawn the ``odor_gaden_rt`` server subprocess. If
            False, only the env is built and the scene is exported (dry run;
            no ROS) — this is the Phase 4.5a path.
        connect_only: attach to an already-running server instead of spawning
            one (implies ``auto_start_gaden``; the session will not stop it).
        bridge: create + connect a :class:`GadenBridge`. If False, no ppm.
        export: export the GADEN scene from ``env.scene_builder`` (single source
            of truth). Turn off only to reuse an existing scene verbatim.
        server_log_dir: directory for the server's captured log.
        wait_timeout: seconds to wait for the server to report ready and for the
            ``/odor_value`` service to appear.
        step_on_timer: run the server free-running instead of lockstep (advanced;
            breaks bridge lockstep — leave False for data collection).
        publish_markers: enable RViz markers on the server (off by default).
        odor_monitor: live ppm UX — ``False`` (off), ``True`` (log + plot),
            ``"log"``, ``"plot"``, or a dict (``log``, ``plot``, ``history_s``,
            ``log_every_n``). Requires an active bridge.
        **env_kwargs: forwarded to the env constructor (e.g. ``robots``,
            ``has_renderer``, ``control_freq``, ``enose_site_offset``).

    Returns:
        An :class:`OdorCosimSession`. Use it as a context manager so the server
        and env are always torn down.
    """
    spec = get_task_spec(env)
    effective_recipe = recipe if recipe is not None else spec.get("default_recipe")
    config_dir = resolve_scenario(scenario, scenario_config)
    scene_id = scene_id or _default_scene_id(env, effective_recipe)

    inner_env = make_env(
        env,
        recipe=effective_recipe,
        scenario_config_dir=config_dir,
        **env_kwargs,
    )

    scene_path = None
    server = None
    owns_server = False
    gaden_bridge = None
    monitor = None
    try:
        if export:
            scene_path = inner_env.scene_builder.export_gaden_scene(config_dir, scene_id=scene_id)

        want_server = auto_start_gaden or connect_only
        if not want_server:
            # Phase 4.5a dry run: env + exported scene only, no ROS.
            return OdorCosimSession(
                inner_env, scene_path=scene_path, scene_id=scene_id, config_dir=config_dir
            )

        # Both the server subprocess and the in-process rclpy bridge open a ROS
        # log file at init; make sure that directory is writable first.
        _ensure_writable_ros_log_dir()

        if not connect_only:
            from odor_sim.runtime.gaden_server import GadenServerManager

            server = GadenServerManager(
                config_dir,
                scene_id=scene_id,
                step_on_timer=step_on_timer,
                publish_markers=publish_markers,
                log_dir=server_log_dir,
            )
            server.start(timeout=wait_timeout)
            owns_server = True

        if bridge:
            from odor_sim.bridge import GadenBridge

            gaden_bridge = GadenBridge()
            if not gaden_bridge.wait_for_server(timeout=wait_timeout):
                raise RuntimeError(
                    "/odor_value not available after starting odor_gaden_rt. "
                    + (
                        "Is a server actually running (connect_only=True)?"
                        if connect_only
                        else f"See server log: {getattr(server, 'log_path', None)}"
                    )
                )

        if odor_monitor and gaden_bridge is not None:
            monitor = _build_odor_monitor(odor_monitor, _gas_types_from_env(inner_env))

        return OdorCosimSession(
            inner_env,
            bridge=gaden_bridge,
            server=server,
            scene_path=scene_path,
            scene_id=scene_id,
            config_dir=config_dir,
            owns_server=owns_server,
            odor_monitor=monitor,
        )
    except Exception:
        # Tear down anything already brought up so a failed make() never leaks a
        # server subprocess or an rclpy node.
        if gaden_bridge is not None:
            try:
                gaden_bridge.close()
            except Exception:  # noqa: BLE001
                pass
        if monitor is not None:
            try:
                monitor.close()
            except Exception:  # noqa: BLE001
                pass
        if server is not None and owns_server:
            try:
                server.stop()
            except Exception:  # noqa: BLE001
                pass
        try:
            inner_env.close()
        except Exception:  # noqa: BLE001
            pass
        raise
