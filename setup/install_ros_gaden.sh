#!/usr/bin/env bash
#
# install_ros_gaden.sh
# ---------------------------------------------------------------------------
# Reproducible setup of the ROS 2 / GADEN half of this project on
# Ubuntu 24.04 (Noble) with ROS 2 Jazzy. Derived from the verified
# setup_gaden.sh; adapted to this repo's layout:
#
#   <repo>/gaden          GADEN (cloned + colcon-built IN PLACE; gitignored)
#   <repo>/ros2_ws/src    OUR ROS 2 packages (overlay; built against gaden)
#
# Idempotent: re-running skips work that is already done (clones, apt, etc.).
#
# Usage:
#   bash setup/install_ros_gaden.sh
#   ROS_MIRROR=http://packages.ros.org/ros2/ubuntu bash setup/install_ros_gaden.sh  # outside CN
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------- locate repo root (this script lives in <repo>/setup) -------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GADEN_DIR="$REPO_ROOT/gaden"
OVERLAY_WS="$REPO_ROOT/ros2_ws"
GADEN_URL="https://github.com/MAPIRlab/gaden.git"
GADEN_BRANCH="jazzy"
OLFACTION_URL="https://github.com/MAPIRlab/olfaction_msgs.git"
GLM_URL="https://github.com/g-truc/glm.git"
ROS_DISTRO_NAME="jazzy"
ROS_MIRROR="${ROS_MIRROR:-https://mirrors.ustc.edu.cn/ros2/ubuntu}"

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
export DEBIAN_FRONTEND=noninteractive
echo "==> repo root : $REPO_ROOT"
echo "==> gaden dir : $GADEN_DIR"
echo "==> overlay   : $OVERLAY_WS"
echo "==> ROS mirror: $ROS_MIRROR"

# ---------------- 1. ROS 2 apt repository + key ----------------
if [ ! -f /etc/apt/sources.list.d/ros2.list ]; then
  $SUDO apt-get update
  $SUDO apt-get install -y curl gnupg lsb-release software-properties-common
  $SUDO add-apt-repository -y universe
  $SUDO install -d -m 0755 /usr/share/keyrings
  if [ ! -s /usr/share/keyrings/ros-archive-keyring.gpg ]; then
    curl -fsSL --retry 3 https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      | $SUDO tee /usr/share/keyrings/ros-archive-keyring.gpg >/dev/null
  fi
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
$ROS_MIRROR $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
    | $SUDO tee /etc/apt/sources.list.d/ros2.list >/dev/null
  $SUDO apt-get update
else
  echo "==> ROS apt source already configured; skipping."
fi

# ---------------- 2. ROS 2 Jazzy + toolchain + gaden_core C++ libs ----------------
# ros-jazzy-ament-cmake-auto is REQUIRED but not declared in gaden's package.xml.
if ! ls /opt/ros/${ROS_DISTRO_NAME}/setup.bash >/dev/null 2>&1; then
  $SUDO apt-get install -y \
    ros-${ROS_DISTRO_NAME}-desktop \
    ros-${ROS_DISTRO_NAME}-ament-cmake-auto \
    ros-dev-tools python3-colcon-common-extensions \
    libfmt-dev zlib1g-dev libboost-dev build-essential cmake git python3-venv
else
  echo "==> ROS ${ROS_DISTRO_NAME} already installed; ensuring build deps present."
  $SUDO apt-get install -y \
    ros-${ROS_DISTRO_NAME}-ament-cmake-auto \
    ros-dev-tools python3-colcon-common-extensions \
    libfmt-dev zlib1g-dev libboost-dev build-essential cmake git python3-venv
fi

# ---------------- 3. gaden source (jazzy) + submodules ----------------
git config --global url."https://github.com/".insteadOf "git@github.com:" || true
if [ ! -d "$GADEN_DIR/.git" ]; then
  git clone --branch "$GADEN_BRANCH" "$GADEN_URL" "$GADEN_DIR"
fi
cd "$GADEN_DIR"
git checkout "$GADEN_BRANCH"
# Fetch the four submodules we need NON-recursively (DDA/third_party/glm has no
# url in .gitmodules and would abort a recursive update).
git submodule update --init gaden_common/third_party/gaden_core
git -C gaden_common/third_party/gaden_core submodule update --init \
  third_party/yaml-cpp third_party/compute third_party/libbsc third_party/DDA

# ---------------- 3b. GLM (needed by DDA, NOT a registered submodule) ----------------
GLM_DIR="$GADEN_DIR/gaden_common/third_party/gaden_core/third_party/DDA/third_party/glm"
if [ ! -f "$GLM_DIR/glm/common.hpp" ]; then
  rm -rf "$GLM_DIR"
  git clone --depth 1 "$GLM_URL" "$GLM_DIR"
fi

# ---------------- 4. olfaction_msgs ----------------
if [ ! -d "$GADEN_DIR/olfaction_msgs/.git" ]; then
  git clone --depth 1 "$OLFACTION_URL" "$GADEN_DIR/olfaction_msgs"
fi

# ---------------- 5. Disable the GUI (idempotent) ----------------
CM="$GADEN_DIR/gaden_common/CMakeLists.txt"
PX="$GADEN_DIR/gaden_common/package.xml"
sed -i -E 's|^([[:space:]]*)include\(cmake/gui\.cmake\)|\1# include(cmake/gui.cmake)  # GUI disabled by install_ros_gaden.sh|' "$CM"
sed -i -E 's|^([[:space:]]*)<depend>ament_imgui</depend>|\1<!-- <depend>ament_imgui</depend>  GUI disabled by install_ros_gaden.sh -->|' "$PX"

# ---------------- 6. Build gaden in place (conda-safe) ----------------
if [ -n "${CONDA_PREFIX:-}" ]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || echo "")"
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"; conda deactivate || true
  fi
fi
# ROS/colcon setup scripts reference unset vars; relax nounset while sourcing.
set +u
# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
set -u
cd "$GADEN_DIR"
colcon build --symlink-install --packages-select \
  olfaction_msgs \
  gaden_common gaden_environment gaden_filament_simulator gaden_msgs \
  gaden_player gaden_preprocessing simulated_anemometer \
  simulated_gas_sensor simulated_tdlas test_env

# ---------------- 7. Build OUR overlay (sources gaden install) ----------------
set +u
# shellcheck disable=SC1091
source "$GADEN_DIR/install/setup.bash"
set -u
# Only build if the overlay actually contains buildable ROS packages
# (package.xml). Before Phase 2 it is an empty placeholder -> skip cleanly.
if find "$OVERLAY_WS/src" -name package.xml 2>/dev/null | grep -q .; then
  cd "$OVERLAY_WS"
  colcon build --symlink-install
  echo "==> Overlay (ros2_ws) built."
else
  echo "==> Overlay (ros2_ws/src) has no ROS packages yet; skipping overlay build."
fi

# ---------------- 8. Convenience: auto-source in every new shell ----------------
BASHRC="${BASHRC:-$HOME/.bashrc}"
BLOCK_MARK='# --- OdorSim / GADEN / ROS 2 Jazzy ---'
if ! grep -qF "$BLOCK_MARK" "$BASHRC" 2>/dev/null; then
  {
    echo ""
    echo "$BLOCK_MARK"
    echo "source /opt/ros/${ROS_DISTRO_NAME}/setup.bash"
    echo "source ${GADEN_DIR}/install/setup.bash"
    echo "[ -f ${OVERLAY_WS}/install/setup.bash ] && source ${OVERLAY_WS}/install/setup.bash"
  } >> "$BASHRC"
  echo "==> Added source block to $BASHRC"
else
  echo "==> $BASHRC already has the OdorSim source block."
fi

cat <<EOF

==================================================================
ROS 2 + GADEN setup complete.

Smoke test (headless, no RViz):
    source /opt/ros/${ROS_DISTRO_NAME}/setup.bash
    source ${GADEN_DIR}/install/setup.bash
    ros2 launch test_env gaden_preproc_launch.py

Recommended one-time (stops conda base shadowing system Python):
    conda config --set auto_activate_base false
==================================================================
EOF
