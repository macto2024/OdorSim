# OdorSim

Co-simulation of **GADEN** (3D odor/gas dispersion) with **robosuite 1.5.2**
(MuJoCo physics) over **ROS 2 Jazzy**, for collecting odor-aware manipulation
demonstrations and training/evaluating VLA policies that fuse an e-nose sense.

See [docs/gaden_robosuite_cosim_d067d61f.plan.md](docs/gaden_robosuite_cosim_d067d61f.plan.md)
for the full design and phase plan.

## Repository layout

```
OdorSim/
├── setup/                     # reproducible install scripts
│   ├── install_ros_gaden.sh   #   ROS 2 + GADEN (clones & builds gaden/ in place)
│   ├── install_sim_env.sh     #   Python venv: robosuite 1.5.2 + mujoco + lerobot
│   ├── requirements-sim.txt   #   sim-side pip deps
│   └── activate.sh            #   `source` this to enter the full environment
├── ros2_ws/src/               # OUR ROS 2 packages (overlay on gaden)
│   └── odor_gaden_rt/         #   real-time moving-source GADEN server (Phase 2)
├── odor_sim/                  # OUR Python package (pip install -e .)
│   ├── envs/                  #   robosuite tasks, OdorObject/OdorProfile, RM65 (Phase 3)
│   ├── bridge/                #   rclpy client + teleop app (Phase 4)
│   ├── sensors/               #   MOX/PID e-nose model, ppm->voltage (Phase 4)
│   ├── recording/             #   LeRobot dataset writer (Phase 5)
│   ├── policy/                #   PolicyAdapter + eval harness (Phase 6)
│   └── config/                #   VOC recipe table, frame map, scenario ids
├── scenarios/                 # GADEN room configs we author
├── docs/                      # design plan
├── gaden/                     # cloned + built by setup (gitignored, not ours)
└── pyproject.toml
```

`gaden/`, `ros2_ws/build|install|log`, `.venv/`, and the deprecated
`simulation/` playground are gitignored.

## Setup (fresh machine)

Ubuntu 24.04 + system Python 3.12. Run in order:

```bash
# 1. ROS 2 Jazzy + GADEN (clones gaden/, builds it, then our overlay)
bash setup/install_ros_gaden.sh

# 2. robosuite/mujoco/lerobot venv (system-site-packages so rclpy is visible)
bash setup/install_sim_env.sh
```

Then, in any new shell:

```bash
source setup/activate.sh   # ROS 2 + GADEN + overlay + venv, in one step
```

## Status

Phases 0-1 complete (repo skeleton + setup scripts). Remaining phases build the
real-time GADEN server, robosuite odor tasks, the teleop/bridge, dataset
recording, and the training-ready eval hooks. Each phase is delivered with a
self-contained test procedure for verification before moving on.
