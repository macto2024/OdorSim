"""Spawn + supervise the ``odor_gaden_rt`` C++ server as a subprocess.

This is the Phase 4.5b piece of the unified ``make()`` facade. It runs::

    ros2 run odor_gaden_rt rt_server --ros-args \
        -p scenarioPath:=<config_dir> -p sceneID:=<scene_id> ...

as a child process, waits until the node logs ``odor_gaden_rt ready.`` (and,
optionally, until the ``/odor_value`` service is up), and tears it down cleanly
on :meth:`stop`. The child inherits the current environment, so ``make()`` must
be called from a shell that has sourced ``setup/activate.sh`` (ROS + GADEN
overlay on ``PATH``). ``ROS_LOG_DIR`` is redirected to a writable directory so
the node can start under a sandbox / restricted ``$HOME``.

One server per session; parallel/vectorized envs are out of scope for v1.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_READY_MARKER = "odor_gaden_rt ready"


class GadenServerManager:
    """Manages the lifecycle of one ``odor_gaden_rt`` process.

    Args:
        scenario_path: GADEN environment configuration directory (contains
            ``config.yaml``); passed as ``scenarioPath``.
        scene_id: which scene (source list) to load; passed as ``sceneID``.
        step_on_timer: run free-running on a wall timer instead of lockstep.
            ``make()`` uses lockstep (False) so the bridge drives ``/gaden/step``.
        publish_markers: publish RViz filament/source markers (off by default;
            unnecessary overhead for headless data collection).
        log_dir: directory for the captured stdout/stderr log. Defaults to
            ``<repo>/.roslog``.
        ros_log_dir: value for ``ROS_LOG_DIR`` in the child env (writable path).
            Defaults to ``<log_dir>/rcl``.
        extra_ros_args: additional ``--ros-args`` tokens appended verbatim.
    """

    def __init__(
        self,
        scenario_path: "str | Path",
        scene_id: str = "scene1",
        *,
        step_on_timer: bool = False,
        publish_markers: bool = False,
        log_dir: "str | Path | None" = None,
        ros_log_dir: "str | Path | None" = None,
        extra_ros_args: "list[str] | None" = None,
    ):
        self.scenario_path = Path(scenario_path).resolve()
        self.scene_id = scene_id
        self.step_on_timer = bool(step_on_timer)
        self.publish_markers = bool(publish_markers)
        self.extra_ros_args = list(extra_ros_args or [])

        self.log_dir = Path(log_dir) if log_dir else _REPO_ROOT / ".roslog"
        self.ros_log_dir = Path(ros_log_dir) if ros_log_dir else self.log_dir / "rcl"

        self._proc: "subprocess.Popen | None" = None
        self._log_fh = None
        self.log_path: "Path | None" = None

    # ------------------------------------------------------------------ #
    def _command(self) -> list[str]:
        cmd = [
            "ros2", "run", "odor_gaden_rt", "rt_server",
            "--ros-args",
            "-p", f"scenarioPath:={self.scenario_path}",
            "-p", f"sceneID:={self.scene_id}",
            "-p", f"stepOnTimer:={'true' if self.step_on_timer else 'false'}",
            "-p", f"publishMarkers:={'true' if self.publish_markers else 'false'}",
        ]
        cmd.extend(self.extra_ros_args)
        return cmd

    def start(self, timeout: float = 60.0) -> None:
        """Spawn the server and block until it logs that it is ready.

        Raises:
            RuntimeError: the process exited early or did not report ready
                within ``timeout`` seconds (the captured log is referenced).
        """
        if self.alive:
            raise RuntimeError("GadenServerManager.start() called twice")

        if not (self.scenario_path / "config.yaml").is_file():
            raise FileNotFoundError(f"No config.yaml in scenarioPath {self.scenario_path}")

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ros_log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"rt_server_{self.scene_id}_{int(time.time())}.log"

        env = dict(os.environ)
        env["ROS_LOG_DIR"] = str(self.ros_log_dir)

        self._log_fh = open(self.log_path, "w+b")
        self._proc = subprocess.Popen(
            self._command(),
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(_REPO_ROOT),
            # own process group so we can signal the whole ros2 launch tree
            start_new_session=True,
        )

        try:
            self._wait_ready(timeout)
        except Exception:
            self.stop()
            raise

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"odor_gaden_rt exited early (code {self._proc.returncode}). "
                    f"See log: {self.log_path}\n{self._tail()}"
                )
            if _READY_MARKER in self._read_log():
                return
            time.sleep(0.2)
        raise RuntimeError(
            f"odor_gaden_rt did not report ready within {timeout:.0f}s. "
            f"See log: {self.log_path}\n{self._tail()}"
        )

    def _read_log(self) -> str:
        if not self.log_path or not self.log_path.exists():
            return ""
        try:
            return self.log_path.read_text(errors="replace")
        except OSError:
            return ""

    def _tail(self, n: int = 20) -> str:
        lines = self._read_log().splitlines()
        return "\n".join(lines[-n:])

    # ------------------------------------------------------------------ #
    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self, timeout: float = 10.0) -> None:
        """Terminate the server: SIGTERM the process group, then SIGKILL."""
        if self._proc is not None:
            if self._proc.poll() is None:
                self._signal_group(signal.SIGTERM)
                try:
                    self._proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self._signal_group(signal.SIGKILL)
                    try:
                        self._proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        pass
            self._proc = None
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            finally:
                self._log_fh = None

    def _signal_group(self, sig) -> None:
        try:
            os.killpg(os.getpgid(self._proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            try:
                self._proc.send_signal(sig)
            except ProcessLookupError:
                pass

    # ------------------------------------------------------------------ #
    def __enter__(self) -> "GadenServerManager":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
