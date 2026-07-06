#!/usr/bin/env bash
#
# install_sim_env.sh
# ---------------------------------------------------------------------------
# Create the Python 3.12 simulation environment for the robosuite side.
#
# Strategy (locked): a venv created with --system-site-packages over the
# SYSTEM python3 (which ROS 2 Jazzy's rclpy is compiled against), so that once
# ROS is sourced, `import rclpy` and `import robosuite` both work in one env.
#
# Run AFTER install_ros_gaden.sh. Usage:
#   bash setup/install_sim_env.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"

echo "==> repo root : $REPO_ROOT"
echo "==> venv      : $VENV_DIR"

# ---------------- 0. sanity: system python must be 3.12 (matches ROS rclpy) --
SYS_PY="/usr/bin/python3"
PYVER="$("$SYS_PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "==> system python: $PYVER ($SYS_PY)"
if [ "$PYVER" != "3.12" ]; then
  echo "!! Expected system Python 3.12 (ROS 2 Jazzy rclpy ABI). Found $PYVER." >&2
  echo "!! Continuing, but rclpy import may fail." >&2
fi

# conda can shadow the system python; make sure it is not active.
if [ -n "${CONDA_PREFIX:-}" ]; then
  echo "!! A conda env is active ($CONDA_PREFIX). Deactivate it before running." >&2
  echo "!! Run: conda deactivate   (and: conda config --set auto_activate_base false)" >&2
  exit 1
fi

# ---------------- 1. create the venv (system site packages) ------------------
if [ ! -d "$VENV_DIR" ]; then
  "$SYS_PY" -m venv --system-site-packages "$VENV_DIR"
  echo "==> created venv."
else
  echo "==> venv already exists; reusing."
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

# ---------------- 2. install sim deps ---------------------------------------
python -m pip install -r "$SCRIPT_DIR/requirements-sim.txt"

# our package (editable)
python -m pip install -e "$REPO_ROOT"

# ---------------- 3. smoke test ---------------------------------------------
echo "==> smoke test (sourcing ROS so rclpy is visible) ..."
if [ -f "$ROS_SETUP" ]; then
  # shellcheck disable=SC1091
  source "$ROS_SETUP"
else
  echo "!! $ROS_SETUP not found - run install_ros_gaden.sh first for rclpy." >&2
fi

python - <<'PY'
import importlib, sys
ok = True
for mod in ["numpy", "mujoco", "robosuite", "odor_sim"]:
    try:
        m = importlib.import_module(mod)
        print(f"  [ok] {mod} {getattr(m, '__version__', '')}")
    except Exception as e:
        ok = False
        print(f"  [FAIL] {mod}: {e}")
# rclpy is only importable when ROS is sourced (PYTHONPATH); report separately.
try:
    import rclpy  # noqa: F401
    print("  [ok] rclpy (ROS sourced)")
except Exception as e:
    print(f"  [warn] rclpy not importable in this shell: {e}")
    print("         -> source /opt/ros/%s/setup.bash before running the bridge." % "jazzy")
try:
    import robosuite
    print(f"  [ok] robosuite version = {robosuite.__version__}")
except Exception:
    pass
sys.exit(0 if ok else 1)
PY

cat <<EOF

==================================================================
Sim env ready at: $VENV_DIR

To use it (single command provided):
    source setup/activate.sh

That sources ROS 2 + GADEN + the overlay, then activates the venv, so
both rclpy and robosuite are importable together.
==================================================================
EOF
