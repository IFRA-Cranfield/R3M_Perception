# r3m_perception
R3M perception ROS2 package.
Now includes the Lamination Sheet 6D pose detection ROS2 package and Logitech C270 webcam ROS2 interface.

## Getting Started

### Dependencies

1. Install Ubuntu 20.04: https://ubuntu.com/tutorials/install-ubuntu-desktop

3. Search for Nvidia GPU driver (Nvidia GPU required): https://www.nvidia.com/download/index.aspx

4. Go to Software & Updates in Ubuntu, select Additional Drivers section and select one approate driver from the list. Than click update. For RTX3050, choose Using NVIDIA driver nvidia-driver-515(proprietary,tested).

5. Check your recommanded CUDA version: 
```
nvidia-smi
```

6. Download and install the CUDA toolkit: https://developer.nvidia.com/cuda-toolkit-archive (CUDA 11.7)

7. Install CUDNN: https://developer.nvidia.com/cudnn (Download cuDNN v8.5.0 (August 8th, 2022), for CUDA 11.x)

8. Install ROS2 Humble following: [ROS2 Humble Tutorials - Installation](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)

8. Install Pytorch: 
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

9. Git clone Megapose6d:
```
git clone https://github.com/megapose6d/megapose6d.git
cd megapose6d && git submodule update --init

```

10. Install Python packages:
```
pip install meshcat selenium omegaconf simplejson line_profiler  opencv-python torchnet tqdm lxml transforms3d panda3d joblib xarray  pandas matplotlib bokeh==2.4.3 plyfile  trimesh  ipdb  panda3d-gltf colorama pyyaml ipykernel scipy pypng h5py  seaborn kornia  pyarrow  dt_apriltags  open3d structlog imageio progressbar pyyaml psutil webdataset opencv-contrib-python  roma  torchgeometry bpy==3.6.0
```

11. Install rclone & pinocchio:
```
sudo apt install rclone
python3 -m pip install pin
```

12. Modify the config file:
In `line 44` of `directory-to-megapose6d/src/megapose/config.py`, change to `PYTHON_BIN_PATH = "/usr/bin/python3"`

14. Download pretrained Megapose models:
```
cd ~/megapose6d/src
python3 -m megapose.scripts.download --megapose_models
```

14. Download example data:
```
python3 -m megapose.scripts.download --example_data
```

15. Testrun the Megapose6d:
```
python3 -m megapose.scripts.run_inference_on_example barbecue-sauce --run-inference
python3 -m megapose.scripts.run_inference_on_example barbecue-sauce --vis-detections
```

16. Build customized data:
Move the `test-lam5` file to `directory-to-megapose6d/local_data/examples`

17. Git clone OWL_ViT:
```
git clone https://github.com/google-research/scenic.git
cd ~/scenic
```
Delect `@master` in line 83 in the setup.py in ~/scenic.
```
python -m pip install -vq .
python -m pip install -r scenic/projects/owl_vit/requirements.txt
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

18. Install big_vision:
```
mkdir /big_vision
git clone https://github.com/google-research/big_vision.git /big_vision
python -m pip install -r /big_vision/big_vision/requirements.txt
```

19. Install ezdxf:
```
pip install ezdxf
```

20. Install R3M_Cell following: https://github.com/R3M-UK/R3M_Cell
    

### Installation
* Download the package in your ROS2 workspace:
```
cd ~/dev_ws/src
git clone https://github.com/YueYaoUoS/r3m_perception.git
cd ~/dev_ws
colcon build --packages-select r3m_perception
```

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
