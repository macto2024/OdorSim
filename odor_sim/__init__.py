"""odor_sim: co-simulation glue between robosuite and GADEN for odor-aware VLA data.

Subpackages (populated across build phases):
    envs      - robosuite task authoring, OdorObject/OdorProfile, RM65 robot
    bridge    - rclpy client to the GADEN real-time server + teleop app
    sensors   - MOX/PID e-nose model (ppm -> voltage), shared offline + eval
    recording - LeRobot dataset writer
    policy    - PolicyAdapter interface + closed-loop eval harness
    config    - VOC recipe table, frame map, scenario ids
"""

__version__ = "0.0.1"
