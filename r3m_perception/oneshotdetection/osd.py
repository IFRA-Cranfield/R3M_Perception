#!/usr/bin/python3
import sys
sys.dont_write_bytecode = True

import sys
import os
BV_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception_setup', 'big_vision')
OS_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception_setup', 'scenic')
sys.path.append(BV_PATH)
sys.path.append(OS_PATH)

import jax
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

    # Initialise -> One-Shot Detection MODEL:
    def __init__(self, threshold = 0.55, MODEL='canonical_checkpoint'):
        
        self.config = configs.owl_v2_clip_l14.get_config(init_mode=MODEL)
        self.module = models.TextZeroShotDetectionModule(
            body_configs=self.config.model.body,
            objectness_head_configs=self.config.model.objectness_head,
            normalize=self.config.model.normalize,
            box_bias=self.config.model.box_bias)

        self.variables = self.module.load_variables(self.config.init_from.checkpoint_path)
        self.threshold = threshold
    
    def check_folders(self, directory, folder_list):
        
        # Get the list of actual folders in the directory:
        actual_folders = [f.name for f in os.scandir(directory) if f.is_dir()]

        # Convert both lists to sets for easier comparison:
        folder_list_set = set(folder_list)
        actual_folders_set = set(actual_folders)

        # Find missing folders and extra folders:
        missing_folders = folder_list_set - actual_folders_set
        extra_folders = actual_folders_set - folder_list_set

        # Print the results:
        if missing_folders:
            print("Missing folders:")
            for folder in missing_folders:
                print(f"  {folder}")
            return False
        else:
            return True

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

    def EXECUTE_OSD(self, inputIMG, cad_list, CAMERA,idx):
        
        self.cad_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'cad', 'mesh')
        self.osd_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'cad', 'osd_execution')
        self.img_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'cad', 'osd_execution', 'OSD_Inputs')
        
        check_result = self.check_folders(self.cad_PATH, cad_list)
        if check_result:

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
            # Get a list of all entries in the directory specified
            
            self.root_path = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'cad', 'mesh')
        
            # Filter the list to include only existing directories
            self.folders1 = [folder for folder in cad_list if os.path.isdir(os.path.join(self.root_path, folder))]
            
            # Return the number of folders
            self.query_num = len(self.folders1)
            
            # Display the folders with their corresponding numbers
            self.figName = []

            for i, folder in enumerate(self.folders1, start=1):
                self.figName.append(self.img_PATH + "/OSD_Input_" + folder + ".png")
                
                # RENDER -> COMMENTED:
                # print(f"{i}. {folder}")
                # os.system(f"python3 ~/dev_ws/src/R3M_Perception/r3m_perception/oneshotdetection/render.py {folder}")
                # print(f"{folder} rendered")
            
            source_images = []
            for figName in self.figName:
                self.figName = skimage_io.imread(figName)
                source_images.append(self.prepare_image(self.figName))
            
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
            source_image_combined = np.expand_dims(source_images[0], axis=0)
            first_flag = True
            
            for source_image in source_images:
                if first_flag:
                    first_flag = False
                else:
                    source_image = np.expand_dims(source_image, axis=0)
                    source_image_combined = np.concatenate((source_image_combined, source_image), axis=0)
            feature_map = image_embedder(source_image_combined)

            b, h, w, d = feature_map.shape
            image_features = feature_map.reshape(b, h * w, d)

            objectnesses = objectness_predictor(image_features)['objectness_logits']

            source_boxes = box_predictor(
                image_features=image_features, feature_map=feature_map
            )['pred_boxes']

            source_class_embeddings = class_predictor(image_features=image_features)[
                'class_embeddings'
            ]
            combined_query_embedding = []
            objectnesses_batch = objectnesses
            source_boxes_batch = source_boxes
            source_class_embeddings_batch = source_class_embeddings
            
            for batch_num in range(self.query_num):
                # Remove batch dimension
                objectnesses = np.array(objectnesses_batch[batch_num])
                source_boxes = np.array(source_boxes_batch[batch_num])
                source_class_embeddings = np.array(source_class_embeddings_batch[batch_num])

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
                query_object_index = index  
                query_embedding = source_class_embeddings[query_object_index]
                combined_query_embedding.append(query_embedding)
                
            combined_query_embedding = np.vstack(combined_query_embedding)
            """## Get predictions for target image with the query embedding"""

            feature_map = image_embedder(target_image[None, ...])

            b, h, w, d = feature_map.shape
            target_boxes = box_predictor(
                image_features=feature_map.reshape(b, h * w, d), feature_map=feature_map
            )['pred_boxes']

            target_class_predictions = class_predictor(
                image_features=feature_map.reshape(b, h * w, d),
                # query_embeddings=query_embedding[None, None, ...],  # [batch, queries, d]
                query_embeddings=combined_query_embedding[None, ...]
            )

            target_boxes = np.array(target_boxes[0])

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(target_image, extent=(0, 1, 1, 0))
            ax.set_axis_off()
            colors = plt.cm.viridis(np.linspace(0, 1, self.query_num))
            
            OSD_RESULT = {}
            RESULT = []
            
            for color,CAD_name,query_num in zip(colors, self.folders1,range(self.query_num)):
                # Remove batch dimension and convert to numpy:

                target_logits = np.array(target_class_predictions['pred_logits'][:,:,query_num])
                target_logits = target_logits.reshape(-1,1)
                #print(np.shape(target_logits))
                top_ind = np.argpartition(target_logits[:,0],-10)[-10:]
                #print(top_ind)
                print(CAD_name)
            
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
                        color = color,
                    )
                    ax.text(
                        cx - w / 2 + 0.015,
                        cy + h / 2 - 0.015,
                        CAD_name + f'Score: {scoreMAX:1.2f}',
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
                #plt.show()
                #print(bbox)
                
                if not bbox:
                    RES = {}
                    RES['Name'] = CAD_name
                    RES['Success'] = False
                    RES['Score'] = 0.0
                    RESULT.append(RES)
                    
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

                    RES = {}
                    RES['Name'] = CAD_name
                    RES['Success'] = True
                    RES['Score'] = scoreMAX.item()
                    RES['tlx'] = TLx
                    RES['tly'] = TLy
                    RES['brx'] = BRx
                    RES['bry'] = BRy
                    RESULT.append(RES)

            outputName = self.osd_PATH + "/OSD_RESULT"+str(idx)+".png"
            fig.savefig(outputName)
            print("")
            print("")
            OSD_RESULT["Result"] = RESULT
            OSD_RESULT["Success"] = True
            return OSD_RESULT
        
        else:
            OSD_RESULT = {}
            OSD_RESULT["Result"] = []
            OSD_RESULT["Success"] = False
            return OSD_RESULT