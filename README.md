# OdorSim

OdorSim connects **GADEN** odor / gas dispersion with **robosuite** manipulation through ROS 2 — for collecting odor-aware demos and exporting datasets.

## Layout

```text
setup/           install and activation scripts
odor_sim/        tasks, bridge, teleop, recording
ros2_ws/         odor_gaden_rt ROS 2 package
scenarios/       GADEN room configs
datasets/        teleop / lerobot outputs
guides/          detailed guides (see below)
```

## Quick start

Ubuntu 24.04. From the repo root:

```bash
bash setup/install_ros_gaden.sh
bash setup/install_sim_env.sh
source setup/activate.sh
```

Smoke test (no GADEN):

```bash
python - <<'EOF'
import odor_sim as odorsim

with odorsim.make(
    "OdorLift", objects=["mango"], auto_start_gaden=False, odor_mode="continuous"
) as cosim:
    obs = cosim.reset()
    print(obs["instruction"], "action_dim=", cosim.action_dim)
    for _ in range(5):
        obs, reward, done, info = cosim.step(cosim.zero_action())
    print("ppm", info.get("ppm"), "enose_voltage", info.get("enose_voltage"))
EOF
```

Collect a teleop demo (live continuous MOX voltage):

```bash
python -m odor_sim.bridge.teleop \
  --env OdorLift --objects milk --device keyboard \
  --odor-mode continuous --odor-monitor
```

Then optionally synthesize other MOX streams and convert to LeRobot — see [guides/lerobot.md](guides/lerobot.md).

## Guides

| Guide | Topic |
|-------|--------|
| [guides/co-simulation.md](guides/co-simulation.md) | Tasks + `odor_sim.make` options |
| [guides/teleop.md](guides/teleop.md) | Recording demos (keyboard / SpaceMouse) |
| [guides/odor-sniff-collection.md](guides/odor-sniff-collection.md) | Stationary odor-only MOX sniffs (ScentFormer) |
| [guides/lerobot.md](guides/lerobot.md) | Synthesize → LeRobot export |
| [guides/gaden-server.md](guides/gaden-server.md) | Manual GADEN RT server |
| [guides/scripted-collection.md](guides/scripted-collection.md) | Scripted `OdorPlace` for OpenVLA-OFT |

Machine-specific install notes: [SETUP.md](SETUP.md), [VNC_SETUP.md](VNC_SETUP.md).

## Tips

- Always `source setup/activate.sh` from the repo root before running commands.
- Prefer `auto_start_gaden=True` (default); use `connect_only=True` only if you started `odor_gaden_rt` yourself.
- Training / eval of VLAs lives in the sibling **Odorbot** repo.
