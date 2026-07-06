# Source me (do NOT execute): `source setup/activate.sh`
# Brings up the full OdorSim environment in the current shell:
#   ROS 2 Jazzy  ->  GADEN install  ->  our overlay  ->  the sim venv
# Order matters: ROS is sourced before the venv so rclpy is on PYTHONPATH.

_odorsim_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_ros_distro="${ROS_DISTRO:-jazzy}"

if [ -n "${CONDA_PREFIX:-}" ]; then
  echo "[activate] warning: conda env active ($CONDA_PREFIX); 'conda deactivate' recommended."
fi

# 1. ROS 2
if [ -f "/opt/ros/${_ros_distro}/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "/opt/ros/${_ros_distro}/setup.bash"
else
  echo "[activate] ROS not found at /opt/ros/${_ros_distro}; run setup/install_ros_gaden.sh"
fi

# 2. GADEN (built in place)
[ -f "$_odorsim_root/gaden/install/setup.bash" ] && \
  source "$_odorsim_root/gaden/install/setup.bash"

# 3. our ROS overlay (exists once Phase 2 is built)
[ -f "$_odorsim_root/ros2_ws/install/setup.bash" ] && \
  source "$_odorsim_root/ros2_ws/install/setup.bash"

# 4. sim venv
if [ -f "$_odorsim_root/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$_odorsim_root/.venv/bin/activate"
else
  echo "[activate] venv not found; run setup/install_sim_env.sh"
fi

echo "[activate] OdorSim env ready (ROS ${_ros_distro} + GADEN + venv)."
