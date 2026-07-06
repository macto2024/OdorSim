# odor_gaden_rt (placeholder)

Our GADEN **real-time** ROS 2 node, implemented in **Phase 2**.

It will wrap GADEN's C++ core (`gaden::Scene` from `RunningSceneMetadata`) to
support **moving gas sources** and on-demand **point concentration queries**:

- sub `/gaden/source_poses` -> mutate each live source's `sourcePosition`
- sub `/gaden/step` -> `scene.AdvanceTimestep()` (lockstep with robosuite)
- serve `/odor_value` (`gaden_msgs/srv/GasPosition`) via `SampleConcentrations`

Built as an overlay that sources `../../gaden/install/setup.bash`
(see `setup/install_ros_gaden.sh`).
