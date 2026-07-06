# odor_gaden_rt

Real-time GADEN server with **moving gas sources** (Phase 2).

GADEN's stock pipeline bakes source positions into pre-saved snapshots, so it
cannot simulate a source the robot picks up. This node wraps GADEN's live core
(`gaden::Scene` over `RunningSimulation`s) instead: sources are mutable at
runtime and concentrations are computed from the live filament cloud.

## Interfaces

| Kind | Name | Type | Purpose |
|------|------|------|---------|
| sub | `/gaden/source_poses` | `geometry_msgs/PoseArray` | pose *i* moves source *i* |
| sub | `/gaden/step` | `std_msgs/Empty` | advance one `deltaTime` (lockstep) |
| pub | `/gaden/sim_time` | `std_msgs/Float32` | current sim time, after each step |
| srv | `/odor_value` | `gaden_msgs/GasPosition` | per-gas ppm at query points |
| srv | `/wind_value` | `gaden_msgs/WindPosition` | wind vector at query points |
| pub | `/gaden/filament_visualization` | `visualization_msgs/Marker` | RViz plume |
| pub | `/gaden/source_visualization` | `visualization_msgs/MarkerArray` | RViz sources |

## Parameters

- `scenarioPath` (required): environment configuration directory
  (e.g. `scenarios/10x6_uniform/environment_configurations/config1`).
  STL geometry and uniform wind are preprocessed in-memory at startup.
- `sceneID` (default `scene1`): which scene (source list) to run.
- `stepOnTimer` (default `false`): free-run at `1/deltaTime` Hz instead of
  waiting for `/gaden/step` (standalone / RViz mode).
- `publishMarkers` (default `true`).

## Run

### Standalone / RViz mode (timer-driven)

```bash
source setup/activate.sh
ros2 run odor_gaden_rt rt_server --ros-args \
  -p scenarioPath:=$PWD/scenarios/10x6_uniform/environment_configurations/config1 \
  -p stepOnTimer:=true
```

### Lockstep mode (default)

In this mode the simulation waits for `/gaden/step` messages and advances one
`deltaTime` per message. Use this when driving the simulator from an external
physics engine (e.g. robosuite).

Terminal 1 — start the server:

```bash
source setup/activate.sh
ros2 run odor_gaden_rt rt_server --ros-args \
  -p scenarioPath:=$PWD/scenarios/10x6_uniform/environment_configurations/config1
```

Terminal 2 — advance time and query:

```bash
# Advance a single step
ros2 topic pub --once /gaden/step std_msgs/msg/Empty '{}'

# Advance N steps (e.g. 20 steps = 1 simulated second with dt=0.05 s)
for i in {1..20}; do
  ros2 topic pub --once /gaden/step std_msgs/msg/Empty '{}'
done

# Advance continuously at 20 Hz
ros2 topic pub /gaden/step std_msgs/msg/Empty '{}' --rate 20
```

Query a concentration or wind vector at any time:

```bash
ros2 service call /odor_value gaden_msgs/srv/GasPosition \
  "{x: [2.5], y: [3.0], z: [0.5]}"

ros2 service call /wind_value gaden_msgs/srv/WindPosition \
  "{x: [2.5], y: [3.0], z: [0.5]}"
```

Monitor simulated time:

```bash
ros2 topic echo /gaden/sim_time
```
