# Run the GADEN RT server

You usually do **not** need to start the server manually. `odor_sim.make(...)` and teleop can export the scene, start `odor_gaden_rt`, connect the bridge, and stop the server for you.

Manual server mode is useful for debugging ROS topics and services.

## Export a scene

```bash
source setup/activate.sh

python -m odor_sim.bridge.export_scene \
  --config-dir scenarios/10x6_uniform/environment_configurations/config1 \
  --scene-id rt_scene \
  --recipe ripe_fruit
```

## Start the server (lockstep)

```bash
ros2 run odor_gaden_rt rt_server --ros-args \
  -p scenarioPath:=$PWD/scenarios/10x6_uniform/environment_configurations/config1 \
  -p sceneID:=rt_scene \
  -p stepOnTimer:=false \
  -p publishMarkers:=false
```

### Parameters

- `scenarioPath`: required path to a GADEN environment config directory containing `config.yaml`.
- `sceneID`: scene file to load, default `scene1`.
- `stepOnTimer`: `false` for lockstep control through `/gaden/step`; `true` to free-run on a timer.
- `publishMarkers`: `true` to publish RViz markers; `false` for lighter headless runs.

### Useful ROS interfaces

- Publish source poses to `/gaden/source_poses`.
- Step lockstep simulation with `/gaden/step`.
- Reset the plume with `/gaden/reset`.
- Read sim time from `/gaden/sim_time`.
- Query ppm with `/odor_value`.
- Query wind with `/wind_value`.
