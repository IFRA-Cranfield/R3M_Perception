# r3m_perception
R3M perception ROS2 package.
It now includes the Lamination Sheet 6D pose detection ROS2 package and Logitech C270 webcam ROS2 interface.

## Getting Started

### Excution
1. Run the simulation:
```
ros2 launch r3mcell_moveit2 r3mcell_moveit2.launch.py
```

2. Run the one-shot action server:
```
ros2 run r3m_perception oneshot.py
```

3. Run the perception node:
```
ros2 run r3m_perception perception_test.py
```

4. Visualize the pose:
```
ros2 topic echo /position_TOPIC
```

## Authors

Yue Yao - Yue.Yao@sheffield.ac.uk
Ze Zhang - Ze.Zhang@sheffield.ac.uk
Mikel Bueno Viso - Mikel.Bueno-Viso@cranfield.ac.uk

## Version History

* 1.1
   * Multiple changes to R3M Perception.
* 0.5
   * Switched the methodology to OWL_ViT & megapose6d.(29.11.2023)
* 0.4
    * Added triangulation so the 3D world coordinate is calculated and published as `/position_TOPIC`.(17.04.2023)
* 0.3
    * Changed the input method of position_publisher node from directly reading from camera port to subscribing `/camer_raw` topic. (19.01.2023)
* 0.2
    * Added Logitech C270 webcam ROS2 interface. (18.01.2023)
* 0.1
    * Initial Release. (17.01.2023)

## License

TBD
