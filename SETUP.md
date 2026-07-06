# GADEN Setup — Ubuntu 24.04 + ROS 2 Jazzy

Complete, verified log of setting up GADEN (https://github.com/MAPIRlab/gaden) on this VM.
The consolidated, re-runnable script is **`setup_gaden.sh`** (same directory).

**Status: DONE.** All 11 packages build; the preprocessing node runs cleanly.

## Target / choices
- OS: Ubuntu 24.04.4 LTS (Noble)
- ROS 2 distro: **Jazzy** (`ros-jazzy-desktop`, includes RViz2)
- Workspace: **build in place** at `/root/OdorSim/gaden` (colcon run from there;
  `olfaction_msgs` cloned inside it). To use the standard `src/` layout instead, move the
  repo under `/root/OdorSim/src/gaden` and run `colcon build` from `/root/OdorSim` — it
  behaves identically (colcon discovers packages recursively).
- GUI (`gaden_gui` / Gaden-RT front-end): **skipped** (disabled in CMake + `ament_imgui` dep dropped)
- Git branch: `jazzy`; submodule SSH URLs rewritten to HTTPS.

## Things that bit us (and the fixes)
1. **Slow ROS mirror** — `packages.ros.org` was ~40 kB/s; switched to USTC (~6 MB/s).
2. **Nested submodules empty** — `gaden_core` has nested submodules
   (`yaml-cpp`, `compute`, `DDA`, `libbsc`); fetched with `--recursive`.
3. **GLM missing** — `DDA` needs GLM at `DDA/third_party/glm`, but it is NOT a registered
   submodule (no `.gitmodules`, `third_party/` gitignored). Cloned g-truc/glm in manually.
   Symptom: `fatal error: glm/common.hpp: No such file or directory`.
4. **`ament_cmake_auto` missing** — four gaden packages `find_package(ament_cmake_auto)`
   but don't declare it in `package.xml`; installed `ros-jazzy-ament-cmake-auto`.
   Symptom: `Could not find a package configuration file provided by "ament_cmake_auto"`.
5. **conda shadows system Python** — conda `base` auto-activates; ROS message generation
   then runs conda's python which lacks `em` (empy). Symptom:
   `ModuleNotFoundError: No module named 'em'`. Fix: `conda deactivate` before building.
6. **rosdep init/update timed out** on GitHub (CN network). Used `rosdepc` (fishros mirror);
   it reported everything already satisfied, so the final script skips rosdep and installs
   the verified apt list directly.

---

## Steps

### 1. ROS 2 apt repository + key (USTC mirror)
```bash
sudo add-apt-repository -y universe
sudo apt-get install -y curl gnupg lsb-release
sudo install -d -m 0755 /usr/share/keyrings
curl -fsSL --retry 3 https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  | sudo tee /usr/share/keyrings/ros-archive-keyring.gpg >/dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
https://mirrors.ustc.edu.cn/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt-get update
```

### 2. Install ROS 2 Jazzy + toolchain + C++ libs (+ ament_cmake_auto)
```bash
sudo apt-get install -y \
  ros-jazzy-desktop ros-jazzy-ament-cmake-auto \
  ros-dev-tools python3-colcon-common-extensions \
  libfmt-dev zlib1g-dev libboost-dev build-essential cmake git
```
Verify: `source /opt/ros/jazzy/setup.bash && echo $ROS_DISTRO` → `jazzy`.

### 3. gaden source (jazzy) + submodules
```bash
git config --global url."https://github.com/".insteadOf "git@github.com:"
cd /root/OdorSim/gaden            # (already cloned here; else: git clone --branch jazzy <url> gaden)
git checkout jazzy
# NON-recursive on purpose: DDA registers a 'third_party/glm' submodule path with NO url
# in .gitmodules, so `git submodule update --recursive` aborts
# (fatal: No url found for submodule path .../DDA/third_party/glm) and never reaches libbsc.
# Fetch the four we actually need; glm is cloned separately in 3b.
git submodule update --init gaden_common/third_party/gaden_core
git -C gaden_common/third_party/gaden_core submodule update --init \
  third_party/yaml-cpp third_party/compute third_party/libbsc third_party/DDA
```

### 3b. GLM into DDA (required — not a submodule)
```bash
cd /root/OdorSim/gaden/gaden_common/third_party/gaden_core/third_party/DDA/third_party
rm -rf glm && git clone --depth 1 https://github.com/g-truc/glm.git glm
ls glm/glm/common.hpp             # must exist
```

### 4. olfaction_msgs (external dep for simulated_* sensors)
```bash
cd /root/OdorSim/gaden
git clone --depth 1 https://github.com/MAPIRlab/olfaction_msgs.git olfaction_msgs
```

### 5. Disable the GUI
In `gaden_common/CMakeLists.txt` comment out `include(cmake/gui.cmake)`; in
`gaden_common/package.xml` comment out `<depend>ament_imgui</depend>`.

### 6. Build (conda deactivated!)
```bash
cd /root/OdorSim/gaden
conda deactivate 2>/dev/null || true          # critical — see gotcha #5
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select \
  olfaction_msgs gaden_common gaden_environment gaden_filament_simulator gaden_msgs \
  gaden_player gaden_preprocessing simulated_anemometer simulated_gas_sensor \
  simulated_tdlas test_env
```
Optional permanent conda fix: `conda config --set auto_activate_base false`.

**Result:** `Summary: 11 packages finished`. Warnings only (setuptools dash-deprecation;
CMake `libOpenCL.so.1 may be hidden` — CUDA 12.8 is present so gaden_core auto-enabled GPU
acceleration; harmless unless a node fails at runtime resolving OpenCL).

### 7. Smoke test (headless — PASSED)
```bash
cd /root/OdorSim/gaden
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch test_env gaden_preproc_launch.py
```
Output ended with `[Gaden-Preprocessing]: Preprocessing done` and
`process has finished cleanly` — build is functional.

---

## Everyday use (new shell)
`setup_gaden.sh` adds this block to `~/.bashrc`, so a new terminal is ready automatically:
```bash
# --- GADEN / ROS 2 Jazzy ---
source /opt/ros/jazzy/setup.bash
source /root/OdorSim/gaden/install/setup.bash
```
Verified: a fresh shell resolves `ros2` and lists the gaden packages with no manual sourcing.

**Recommended one-time** — a new shell still auto-activates conda `base`
(`python3 → miniconda`), which breaks future `colcon build` (the `em` error) and is a
runtime risk. Turn it off:
```bash
conda config --set auto_activate_base false   # reversible: ... true
```

See `GADEN_tutorial.md` in the repo for the full preprocess → simulate → playback workflow.

