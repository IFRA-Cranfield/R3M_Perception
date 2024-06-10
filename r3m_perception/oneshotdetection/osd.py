#!/usr/bin/python3
import sys
import os
BV_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'installation', 'big_vision')
OS_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'installation', 'scenic')
sys.path.append(BV_PATH)
sys.path.append(OS_PATH)

import jax
from matplotlib import pyplot as plt
import numpy as np
from scenic.projects.owl_vit import configs
from scenic.projects.owl_vit import models
from scipy.special import expit as sigmoid
import skimage
from skimage import io as skimage_io
from skimage import transform as skimage_transform
import functools

import matplotlib.pyplot as plt
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.properties import Properties, LayoutProperties
# import wx
import glob
import re
import yaml

class OSD_CLASS():

    def __init__(self, CADName, CAMERA, threshold = 0.85, MODEL='canonical_checkpoint'):
        self.config = configs.owl_v2_clip_l14.get_config(init_mode=MODEL)
        self.module = models.TextZeroShotDetectionModule(
            body_configs=self.config.model.body,
            objectness_head_configs=self.config.model.objectness_head,
            normalize=self.config.model.normalize,
            box_bias=self.config.model.box_bias)

        self.variables = self.module.load_variables(self.config.init_from.checkpoint_path)
        self.threshold = threshold
        #INITIALISE the OSD MODEL:
        self.cad_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'cad', CADName)
        self.img_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'cad', CADName, 'osd_execution')

        #INITIALISE the CAMERA model:
        camera_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'config', CAMERA)
        CAM = camera_PATH + "/config.yaml"

        #Load resolution from yaml:
        # Get RECIPE VALUES:
        with open(CAM, 'r') as YAML:
            camYAML = yaml.safe_load(YAML)
            
        self.w = camYAML["resolution"]["W"] 
        self.h = camYAML["resolution"]["H"] 

        #RENDER.py script:
        # CADName variable as input.
        # cad_PATH
        # img_PATH
        # same as above.
        os.system("python3 ~/dev_ws/src/R3M_Perception/r3m_perception/oneshotdetection/render.py " + CADName)
        print("rendered")
        self.figName = self.img_PATH + "/OSD_Input.png"
        
    def prepare_image(self, name):
        
        # Load example image:
        # image_uint8 = skimage_io.imread(name)
        image_uint8 = name[:,:,:3]
        image = image_uint8.astype(np.float32) / 255.0

        # Pad to square with gray pixels on bottom and right:
        h, w, _ = image.shape
        size = max(h, w)
        image_padded = np.pad(
            image, ((0, size - h), (0, size - w), (0, 0)), constant_values=0.5
        )

        # Resize to model input size:
        return skimage.transform.resize(
            image_padded,
            (self.config.dataset_configs.input_size, self.config.dataset_configs.input_size),
            anti_aliasing=True,
        )

    def EXECUTE_OSD(self, inputIMG):
         
        self.figName = skimage_io.imread(self.figName)
        source_image = self.prepare_image(self.figName)
        
        # Process INPUT IMAGE:
        inputIMG = inputIMG[0:int(self.h),int((self.w - self.h/3*4)/2):int(self.w - (self.w - self.h/3*4)/2)]
        target_image = self.prepare_image(inputIMG)
        image_embedder = jax.jit(
            functools.partial(
                self.module.apply, self.variables, train=False, method=self.module.image_embedder
            )
        )

        objectness_predictor = jax.jit(
            functools.partial(
                self.module.apply, self.variables, method=self.module.objectness_predictor
            )
        )

        box_predictor = jax.jit(
            functools.partial(self.module.apply, self.variables, method=self.module.box_predictor)
        )

        class_predictor = jax.jit(
            functools.partial(self.module.apply, self.variables, method=self.module.class_predictor)
        )

        # Embedd images and get boxes, without text queries:
        feature_map = image_embedder(source_image[None, ...])

        b, h, w, d = feature_map.shape
        image_features = feature_map.reshape(b, h * w, d)

        objectnesses = objectness_predictor(image_features)['objectness_logits']

        source_boxes = box_predictor(
            image_features=image_features, feature_map=feature_map
        )['pred_boxes']

        source_class_embeddings = class_predictor(image_features=image_features)[
            'class_embeddings'
        ]

        # Remove batch dimension
        objectnesses = np.array(objectnesses[0])
        source_boxes = np.array(source_boxes[0])
        source_class_embeddings = np.array(source_class_embeddings[0])

        top_k = 1
        objectnesses = sigmoid(objectnesses)
        objectness_threshold = np.partition(objectnesses, -top_k)[-top_k]

        for i, (box, objectness) in enumerate(zip(source_boxes, objectnesses)):
            if objectness < objectness_threshold:
                continue

            cx, cy, w, h = box
            index = i
        #print(index)

        # Get the query embedding with the index of the selected object.
        # We're using the rocket:
        query_object_index = index  # Index of the rocket box above.
        query_embedding = source_class_embeddings[query_object_index]

        """## Get predictions for target image with the query embedding"""

        feature_map = image_embedder(target_image[None, ...])

        b, h, w, d = feature_map.shape
        target_boxes = box_predictor(
            image_features=feature_map.reshape(b, h * w, d), feature_map=feature_map
        )['pred_boxes']

        target_class_predictions = class_predictor(
            image_features=feature_map.reshape(b, h * w, d),
            query_embeddings=query_embedding[None, None, ...],  # [batch, queries, d]
        )

        # Remove batch dimension and convert to numpy:
        target_boxes = np.array(target_boxes[0])
        target_logits = np.array(target_class_predictions['pred_logits'][0])
        #print(np.shape(target_logits))

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(target_image, extent=(0, 1, 1, 0))
        ax.set_axis_off()

        top_ind = np.argpartition(target_logits[:,0],-10)[-10:]
        #print(top_ind)
        bbox = []
        
        scoreMAX = 0
        Found = False
        ELEMENT = None
        for i in top_ind:
            score = sigmoid(target_logits[i, 0])
            if (score > scoreMAX and score > self.threshold):
                #print("SCORE: " + str(score))
                scoreMAX = score
                ELEMENT = i
                Found = True

        if Found: 
            cx, cy, w, h = target_boxes[ELEMENT]
            #print("cx: " + str(cx) + " cy: " + str(cy) + " w: " + str(w) + " h: " + str(h))
            ax.plot(
                [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
                [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2],
                color='lime',
            )
            ax.text(
                cx - w / 2 + 0.015,
                cy + h / 2 - 0.015,
                f'Score: {scoreMAX:1.2f}',
                ha='left',
                va='bottom',
                color='black',
                bbox={
                    'facecolor': 'white',
                    'edgecolor': 'lime',
                    'boxstyle': 'square,pad=.3',
                },
            )
            bbox.append([cx,cy,w,h])
       
        #ax.set_xlim(0, 1)
        #ax.set_ylim(1, 0)
        #ax.set_title(f'One-Shot Detection: RESULT')
        # plt.show()
        
        outputName = self.img_PATH + "/OSD_RESULT.png"
        fig.savefig(outputName)
        #print(bbox)
        result = {}
        if not bbox:
            
            result['score'] = 0.0
            
            print("One-Shot Detection: Executed! Result:")
            print("ERROR: The object could not be detected with enough accuracy.")
            
        else:
            
            cx = bbox[0][0]
            cy = bbox[0][1]
            w = bbox[0][2]
            h = bbox[0][3]
            
            TLx = round((cx*1008 - w*1008 / 2) - 10)
            TLy = round((cy*1008 - h*1008 / 2) - 10)
            BRx = round((cx*1008 + w*1008 / 2) + 10)
            BRy = round((cy*1008 + h*1008 / 2) + 10)
            
            ACCscore = scoreMAX.item()*100.0
        
            print("One-Shot Detection: Executed! Result:")
            print("The object has been detected with an accuracy score of -> " + str(round(ACCscore,2)) + "%")
            print("Bounding Box:")
            print("  - Top left corner pixel (x,y) -> (" + str(TLx) + "," + str(TLy) + ")")
            print("  - Bottom right corner pixel (x,y) -> (" + str(BRx) + "," + str(BRy) + ")")
            
            result['tlx'] = TLx
            result['tly'] = TLy
            result['brx'] = BRx
            result['bry'] = BRy
            result['score'] = scoreMAX.item()

        print("")
        print("")

        return result