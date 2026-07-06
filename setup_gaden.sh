#!/usr/bin/env bash
#
# setup_gaden.sh
# ---------------------------------------------------------------------------
# Reproducible setup of GADEN (https://github.com/MAPIRlab/gaden) on
# Ubuntu 24.04 (Noble) with ROS 2 Jazzy.  Verified end-to-end on a fresh
# Ubuntu 24.04.4 VM: all 11 packages build and the preprocessing node runs.
#
# Scope / choices (see SETUP.md for the full story):
#   * ROS 2 Jazzy desktop (includes RViz2)
#   * gaden built IN PLACE at $GADEN_DIR (colcon run from there)
#   * GUI (gaden_gui / Gaden-RT front-end) is DISABLED
#   * ROS apt packages pulled from the USTC mirror (fast in CN)
#
# Usage:
#   bash setup_gaden.sh                      # defaults to /root/OdorSim
#   WS=/opt/gaden_ws bash setup_gaden.sh     # custom workspace root
#   ROS_MIRROR=http://packages.ros.org/ros2/ubuntu bash setup_gaden.sh   # outside CN
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------- configuration ----------------
WS="${WS:-/root/OdorSim}"                                   # workspace root
GADEN_DIR="$WS/gaden"                                       # gaden repo (built in place)
GADEN_URL="https://github.com/MAPIRlab/gaden.git"
GADEN_BRANCH="jazzy"
OLFACTION_URL="https://github.com/MAPIRlab/olfaction_msgs.git"
GLM_URL="https://github.com/g-truc/glm.git"
ROS_DISTRO_NAME="jazzy"
ROS_MIRROR="${ROS_MIRROR:-https://mirrors.ustc.edu.cn/ros2/ubuntu}"

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
export DEBIAN_FRONTEND=noninteractive
echo "==> Workspace root : $WS"
echo "==> gaden dir      : $GADEN_DIR"
echo "==> ROS mirror     : $ROS_MIRROR"

# ---------------- 1. ROS 2 apt repository + key ----------------
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

# ---------------- 2. ROS 2 Jazzy + toolchain + gaden_core C++ libs ----------------
# NOTE: ros-jazzy-ament-cmake-auto is REQUIRED but NOT declared in gaden's package.xml
#       files (gaden_environment/player/preprocessing/filament_simulator call
#       find_package(ament_cmake_auto) directly). rosdep won't pull it -> install here.
# NOTE: rosdep is intentionally not used; this explicit list is verified complete
#       (rosdep reported "All required rosdeps installed successfully" with nothing to add).
$SUDO apt-get install -y \
  ros-${ROS_DISTRO_NAME}-desktop \
  ros-${ROS_DISTRO_NAME}-ament-cmake-auto \
  ros-dev-tools python3-colcon-common-extensions \
  libfmt-dev zlib1g-dev libboost-dev build-essential cmake git

# ---------------- 3. gaden source (jazzy) + submodules ----------------
# gaden's submodules use git@github.com SSH URLs; rewrite to HTTPS (public repos).
git config --global url."https://github.com/".insteadOf "git@github.com:"
mkdir -p "$WS"
if [ ! -d "$GADEN_DIR/.git" ]; then
  git clone --branch "$GADEN_BRANCH" "$GADEN_URL" "$GADEN_DIR"
fi
cd "$GADEN_DIR"
git checkout "$GADEN_BRANCH"
# gaden_core backend, then its nested submodules EXPLICITLY and NON-recursively.
# Do NOT use --recursive here: DDA registers a submodule path 'third_party/glm'
# but gives NO url for it in .gitmodules, so 'git submodule update --recursive'
# aborts with 'fatal: No url found for submodule path .../DDA/third_party/glm'
# (and never reaches libbsc). We fetch the four we need, then clone glm in 3b.
# (gaden_gui is deliberately left un-fetched; GUI is disabled below.)
git submodule update --init gaden_common/third_party/gaden_core
git -C gaden_common/third_party/gaden_core submodule update --init \
  third_party/yaml-cpp third_party/compute third_party/libbsc third_party/DDA

# ---------------- 3b. GLM (needed by DDA, NOT a registered submodule) ----------------
# DDA has no .gitmodules and gitignores third_party/, so 'submodule --recursive'
# never fetches glm. Without it: fatal error: glm/common.hpp: No such file or directory
GLM_DIR="$GADEN_DIR/gaden_common/third_party/gaden_core/third_party/DDA/third_party/glm"
if [ ! -f "$GLM_DIR/glm/common.hpp" ]; then
  rm -rf "$GLM_DIR"
  git clone --depth 1 "$GLM_URL" "$GLM_DIR"
fi

# ---------------- 4. olfaction_msgs (external dep for simulated_* sensors) ----------------
if [ ! -d "$GADEN_DIR/olfaction_msgs/.git" ]; then
  git clone --depth 1 "$OLFACTION_URL" "$GADEN_DIR/olfaction_msgs"
fi

# ---------------- 5. Disable the GUI (skip gaden_gui / ament_imgui) ----------------
# The jazzy branch makes ament_imgui a hard dependency and compiles gaden_gui.
# We don't build the GUI, so comment out both (idempotent: no-op if already done).
CM="$GADEN_DIR/gaden_common/CMakeLists.txt"
PX="$GADEN_DIR/gaden_common/package.xml"
sed -i -E 's|^([[:space:]]*)include\(cmake/gui\.cmake\)|\1# include(cmake/gui.cmake)  # GUI disabled by setup_gaden.sh|' "$CM"
sed -i -E 's|^([[:space:]]*)<depend>ament_imgui</depend>|\1<!-- <depend>ament_imgui</depend>  GUI disabled by setup_gaden.sh -->|' "$PX"

# ---------------- 6. Build (conda-safe) ----------------
# ROS 2 message generation (rosidl) must run the SYSTEM python (has 'empy'/em),
# NOT conda's python. If a conda env is active, deactivate it first.
if [ -n "${CONDA_PREFIX:-}" ]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || echo "")"
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda deactivate || true
  fi
fi
# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
cd "$GADEN_DIR"
colcon build --symlink-install --packages-select \
  olfaction_msgs \
  gaden_common gaden_environment gaden_filament_simulator gaden_msgs \
  gaden_player gaden_preprocessing simulated_anemometer \
  simulated_gas_sensor simulated_tdlas test_env

# ---------------- 7. Convenience: auto-source GADEN in every new shell ----------------
# Appends the source block to ~/.bashrc (idempotent — skipped if already present).
BASHRC="${BASHRC:-$HOME/.bashrc}"
if ! grep -qF '# --- GADEN / ROS 2 Jazzy ---' "$BASHRC" 2>/dev/null; then
  cat >> "$BASHRC" <<EOF2

# --- GADEN / ROS 2 Jazzy ---
source /opt/ros/${ROS_DISTRO_NAME}/setup.bash
source ${GADEN_DIR}/install/setup.bash
EOF2
  echo "==> Added GADEN/ROS source block to $BASHRC"
else
  echo "==> $BASHRC already contains the GADEN/ROS block; leaving it as is"
fi

# ---------------- done ----------------
cat <<EOF

==================================================================
GADEN build complete.

Your ~/.bashrc now auto-sources ROS 2 + the GADEN workspace, so a NEW shell is
ready to go. (In THIS shell, run the two source lines once, or open a new terminal.)

Smoke test (headless, no RViz):
    ros2 launch test_env gaden_preproc_launch.py

Recommended one-time (stops conda base from shadowing system Python at runtime):
    conda config --set auto_activate_base false
==================================================================
EOF
