#!/usr/bin/python3
import sys
sys.dont_write_bytecode = True

# R3M Project - Cranfield University
# ROS2.0 TEMPLATES - Action Server
import os
MP_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception_setup', 'megapose6d', 'src')

# Import libraries:
from dataclasses import dataclass
@dataclass
class Trans:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot, show, figure
from PIL import Image
import skimage
import cv2
import math
import yaml

from skimage.util import img_as_float

import sys
sys.path.append(MP_PATH)
# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay

from scipy.spatial.transform import Rotation as R

# For zero-shot:
from megapose.zero_shot_interface import load_detections_zero_multi

# For UI
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject
from PyQt5.QtGui import QPixmap
from threading import Thread
logger = get_logger(__name__)

class MEGAPOSE_CLASS():
    
    def __init__(self):
        
        self.parser = argparse.ArgumentParser()
        # self.parser.add_argument("example_name")
        self.parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
        # parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis-icp")
        self.parser.add_argument("--vis-detections", action="store_true")
        self.parser.add_argument("--run-inference", action="store_true")
        self.parser.add_argument("--vis-outputs", action="store_true")
        self.args = self.parser.parse_args()
        # data_dir = os.getenv("MEGAPOSE_DATA_DIR")
        
        data_dir = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception','r3m_perception', 'cad')
        assert data_dir
        self.example_dir = data_dir
        self.object_dataset = self.make_object_dataset(self.example_dir)
        self.model_info = NAMED_MODELS[self.args.model]
        self.pose_estimator = load_named_model(self.args.model, self.object_dataset).cuda()
        
    def EulerToQuat(self, roll, pitch, yaw):

        RESULT = {"qx": None, "qy": None, "qz": None, "qw": None}

        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        RESULT["qx"] = sr * cp * cy - cr * sp * sy
        RESULT["qy"] = cr * sp * cy + sr * cp * sy
        RESULT["qz"] = cr * cp * sy - sr * sp * cy
        RESULT["qw"] = cr * cp * cy + sr * sp * sy

        return(RESULT)


    def matrix_to_xyzrpy(self,matrix):
        x, y, z = matrix[:3, 3]
        rotation_matrix = matrix[:3, :3]
        r = R.from_matrix(rotation_matrix)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        return x, y, z, roll, pitch, yaw


    def my_visual(self,camera_data_v):
        camera_data_v.TWC = Transform(np.eye(4))
        object_datas = self.load_object_data(Path(self.example_dir) / 'm6d_execution' / "outputs" / "object_data.json")

        renderer = Panda3dSceneRenderer(self.object_dataset)

        camera_data_v, object_datas = convert_scene_observation_to_panda3d(self.camera_data, object_datas)
        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((1.0, 1.0, 1.0, 1)),
            ),
        ]
        renderings = renderer.render_scene(
            object_datas,
            [camera_data_v],
            light_datas,
            render_depth=False,
            render_binary_mask=False,
            render_normals=False,
            copy_arrays=True,
        )[0]

        plotter = BokehPlotter()
        contour_overlay = make_contour_overlay(
            self.rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
        )["img"]
        fig_contour_overlay = plotter.plot_image(contour_overlay)
        
        export_png(fig_contour_overlay, filename=self.vis_dir / f"contour_overlay.png")


    def load_object_data(self,data_path: Path) -> List[ObjectData]:
        object_data = json.loads(data_path.read_text())
        object_data = [ObjectData.from_json(d) for d in object_data]
        return object_data


    def make_object_dataset(self,example_dir: Path) -> RigidObjectDataset:
        
        rigid_objects = []
        mesh_units = "mm"
        object_dirs = (Path(example_dir) / "mesh").iterdir()
        
        for object_dir in object_dirs:
            label = object_dir.name
            mesh_path = None
            for fn in object_dir.glob("*"):
                if fn.suffix in {".obj", ".ply"}:
                    assert not mesh_path, f"there multiple meshes in the {label} directory"
                    mesh_path = fn
            assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
            rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
            # TODO: fix mesh units
            
        rigid_object_dataset = RigidObjectDataset(rigid_objects)
        return rigid_object_dataset
    

    def save_predictions(self,
        example_dir: Path,
        pose_estimates: PoseEstimatesType,
    ) -> None:
        labels = pose_estimates.infos["label"]
        poses = pose_estimates.poses.cpu().numpy()
        object_data = [
            ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
        ]
        object_data_json = json.dumps([x.to_json() for x in object_data])
        output_fn = Path(example_dir) / 'm6d_execution' / "outputs" / "object_data.json"
        output_fn.parent.mkdir(exist_ok=True)
        output_fn.write_text(object_data_json)
        logger.info(f"Wrote predictions: {output_fn}")
        return

    def EXECUTE_FI(self, frame, CAMERA, BB):
        
        self.cad_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'cad', 'mesh')
        camera_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'config', CAMERA)
        
        self.camera_data = CameraData.from_json((Path(camera_PATH) / 'camera_data.json').read_text())
        self.vis_dir = Path(self.example_dir) / 'm6d_execution' / 'visualizations'
        self.vis_dir.mkdir(exist_ok=True)
        CAM = camera_PATH + "/config.yaml"
        TRANS = camera_PATH + "/transformation.yaml"
        
        #Load resolution from yaml:
        # Get RECIPE VALUES:
        with open(CAM, 'r') as YAML:
            camYAML = yaml.safe_load(YAML)
            
        self.w = camYAML["resolution"]["W"] 
        self.h = camYAML["resolution"]["H"] 

        with open(TRANS, 'r') as YAML:
            transYAML = yaml.safe_load(YAML)
            
        self.t1 = transYAML["transformation"]["t1"] 
        self.t2 = transYAML["transformation"]["t2"] 
        self.img = frame[0:int(self.h),int((self.w - self.h/3*4)/2):int(self.w - (self.w - self.h/3*4)/2)]
        self.rgb, depth = self.img, None
        self.rgb.shape[:2] == self.camera_data.resolution
        
        # self.bbox -> BB:
        self.bbox_extended = []
        for item in BB['Result']:
            label = item['Name']
            bbox_modal = [
                item['tlx'] * (self.h/756),
                item['tly'] * (self.h/756),
                item['brx'] * (self.h/756),
                item['bry'] * (self.h/756)
            ]
            self.bbox_extended.append({"label": label, "bbox_modal": bbox_modal})
        
        self.detections = load_detections_zero_multi(zero_shot_bbox=self.bbox_extended)
        self.observation = ObservationTensor.from_numpy(self.rgb, depth, self.camera_data.K).cuda()

        self.output, _ = self.pose_estimator.run_inference_pipeline(
            self.observation, detections=self.detections, **self.model_info["inference_parameters"]
        )
        
        print(self.output)
        
        labels = self.output.infos["label"]
        poses = self.output.poses.cpu().numpy()
        print(poses)
        
        self.save_predictions(self.example_dir, self.output)
        self.my_visual(self.camera_data)
        
        M6D_RESULT = []
        
        for pose, label in zip(poses, labels):
            print('Two: ', Transform(pose))
            self.initial_message_received = True
            
            MSG = Trans(0.0,0.0,0.0,0.0,0.0,0.0)
            transformation1 = np.array(self.t1)
            transformation2 = np.array(self.t2)
            # MSG.x, MSG.y, MSG.z, MSG.row, MSG.pitch, MSG.yaw = self.matrix_to_xyzrpy(np.matmul(Transform(poses[0]).matrix, transformation))
            MSG.x, MSG.y, MSG.z, MSG.roll, MSG.pitch, MSG.yaw = self.matrix_to_xyzrpy(np.matmul(transformation1,np.matmul(transformation2,Transform(pose).matrix)))
            
            ORIENTATION = self.EulerToQuat(MSG.roll, MSG.pitch, MSG.yaw)
            
            # RETURN RESULT:
            RESULT = {}
            RESULT['Name'] = label
            RESULT['x'] = round(MSG.x, 5)
            RESULT['y'] = round(MSG.y, 5)
            RESULT['z'] = round(MSG.z, 5)
            RESULT['qx'] = round(ORIENTATION["qx"], 5)
            RESULT['qy'] = round(ORIENTATION["qy"], 5)
            RESULT['qz'] = round(ORIENTATION["qz"], 5)
            RESULT['qw'] = round(ORIENTATION["qw"], 5)
            M6D_RESULT.append(RESULT)
        
        return M6D_RESULT

    def EXECUTE_RTI(self, frame):
        
        self.img = frame[0:int(self.h),int((self.w - self.h/3*4)/2):int(self.w - (self.w - self.h/3*4)/2)]
        self.rgb, depth = self.img, None
        self.rgb.shape[:2] == self.camera_data.resolution
        #self.observation = ObservationTensor.from_numpy(self.rgb, depth, self.camera_data.K).cuda()
        
        self.output, _ = self.pose_estimator.forward_refiner(
            self.observation,
            self.output,
            n_iterations=1,
            keep_all_outputs=True,
            cuda_timer=False,
        )
        
        #labels = self.output.infos["label"]
        
        self.output = self.output[f"iteration={1}"]
        poses = self.output.poses.cpu().numpy()
        labels = self.output.infos["label"]
        
        M6D_RESULT = []
        
        for pose, label in zip(poses, labels):
            print('Two: ', Transform(poses[0]))
            #self.save_predictions(self.example_dir, self.output)
            #self.my_visual(self.camera_data)
            MSG = Trans(0.0,0.0,0.0,0.0,0.0,0.0)
            transformation1 = np.array(self.t1)
            transformation2 = np.array(self.t2)
            # MSG.x, MSG.y, MSG.z, MSG.row, MSG.pitch, MSG.yaw = self.matrix_to_xyzrpy(np.matmul(Transform(poses[0]).matrix, transformation))
            MSG.x, MSG.y, MSG.z, MSG.roll, MSG.pitch, MSG.yaw = self.matrix_to_xyzrpy(np.matmul(transformation1,np.matmul(transformation2,Transform(poses[0]).matrix)))
            
            ORIENTATION = self.EulerToQuat(MSG.roll, MSG.pitch, MSG.yaw)
            
            # RETURN RESULT:
            RESULT = {}
            RESULT['Name'] = label
            RESULT['x'] = round(MSG.x, 5)
            RESULT['y'] = round(MSG.y, 5)
            RESULT['z'] = round(MSG.z, 5)
            RESULT['qx'] = round(ORIENTATION["qx"], 5)
            RESULT['qy'] = round(ORIENTATION["qy"], 5)
            RESULT['qz'] = round(ORIENTATION["qz"], 5)
            RESULT['qw'] = round(ORIENTATION["qw"], 5)
            M6D_RESULT.append(RESULT)

        return M6D_RESULT