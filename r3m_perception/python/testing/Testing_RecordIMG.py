#!/usr/bin/python3
import sys
sys.dont_write_bytecode = True

# # # # # # # # # # # # # # # # # #                                  
#                                 #
#   ===== COPYRIGHT HERE =====    #
#                                 #
# # # # # # # # # # # # # # # # # #

# ========================================================================================= #
# ======================================== INCLUDE ======================================== #
# ========================================================================================= #

# System:
import os, sys, time, xacro, random, yaml

# OpenCV:
import cv2

# ROS2:
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

# IMPORT /SpawnEntity and /DeleteEntity ROS2 Services:
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.srv import DeleteEntity

# R3M Perception PATH:
PATH_P = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception')

# Import FUNCTIONS:
PATH_F = PATH_P + "/python/functions"
sys.path.append(PATH_F)
# IMPORT PythonClass -> Get OBJECT POSES:
from ObjectState import OBJECT
# Import convertIMG:
from convertIMG import toCV2_fromTOPIC

# ========================================================================================= #
# ========================================================================================= #
# Function -> Save IMAGE:
def saveIMG(i, PATH, IMG):
    imgNAME = PATH + "/" + str(i) + ".png"
    cv2.imwrite(imgNAME, IMG)
    print("Image -> " + str(imgNAME) + " saved.")

# ========================================================================================= #
# ========================================================================================= #
# Function -> Record OBJECT POSES:
def recordObjPose(yamlNAME, i, OBJ):
    
    ObjectPoseList = OBJ.GetObjectPose()

    with open(yamlNAME, 'r') as F:
        testINFO = yaml.safe_load(F)

    for x in ObjectPoseList:
        
        POSE = {}
        POSE["x"] = round(x["Pose"].x, 5)
        POSE["y"] = round(x["Pose"].y, 5)
        POSE["z"] = round(x["Pose"].z, 5)
        POSE["qx"] = round(x["Pose"].qx, 5)
        POSE["qy"] = round(x["Pose"].qy, 5)
        POSE["qz"] = round(x["Pose"].qz, 5)
        POSE["qw"] = round(x["Pose"].qw, 5)

        inputDICT = {}
        inputDICT[x["Name"]] = {}
        inputDICT[x["Name"]]["Gazebo"] = POSE

        testINFO[str(i)] = inputDICT

        print("ObjectPose information of object -> " + x["Name"] + " recorded.")

    with open(yamlNAME, 'w') as F:
        yaml.dump(testINFO, F)

# ========================================================================================= #
# ========================================================================================= #
# ServiceClient (SPAWN/DELETE OBJECT):

class EntityClient(Node):

    def __init__(self):

        # Initialise ROS2 Node:
        super().__init__('R3MP_TEST_EntityClient')

        # Create ROS2 Service Clients:
        self.cli_SPAWN = self.create_client(SpawnEntity, "/spawn_entity")  
        self.cli_DELETE = self.create_client(DeleteEntity, "/delete_entity") 

        # Declare REQUEST variable (of CUSTOM DATA type):
        self.req_SPAWN = SpawnEntity.Request()  
        self.req_DELETE = DeleteEntity.Request()

    def spawn_REQUEST(self, ObjectList):
        
        for x in ObjectList:
            
            # LOAD URDF of R3M_OBJECT:
            urdf_file_path = os.path.join(get_package_share_directory('r3m_perception'), 'urdf', 'objects', 'R3MPerceptionOBJ.urdf')
            xacro_file = xacro.process_file(urdf_file_path, mappings={"name": x})
            
            # ARGUMENTS:
            self.req_SPAWN.name = x
            self.req_SPAWN.xml = xacro_file.toxml()
            self.req_SPAWN.initial_pose.position.x = random.uniform(0.5, 0.7)
            self.req_SPAWN.initial_pose.position.y = random.uniform(0.1, 0.9)
            self.req_SPAWN.initial_pose.position.z = 1.0
            # Add here -> Random orientation.

            # Assign RESULT value (future):
            self.future_SPAWN = self.cli_SPAWN.call_async(self.req_SPAWN)

    def delete_REQUEST(self, ObjectList):

        for x in ObjectList:
            self.req_DELETE.name = x
            self.future_DELETE = self.cli_DELETE.call_async(self.req_DELETE)

# ========================================================================================= #
# ========================================= MAIN ========================================== #
# ========================================================================================= #
def main(args=None):
    
    print("")
    print("")
    
    print(" ======================== ")
    print(" R3M Perception - TESTING ")
    print(" RECORD IMAGES from GzSim ")
    print(" ======================== ")
    print("")

    # INPUT VARIABLES:
    FOLDER = os.path.join(os.path.expanduser('~'), 'PerceptionTesting')
    TESTName = "Test001"
    ITERATIONS = 100
    OBJECTS = ['adapter_plate_triangular']
    CAMERA = "lenovoFHD_gazebo"
    
    PATH = FOLDER + "/" + TESTName
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    else:
        print(TESTName + " folder exists. Please delete it or define a new folder name.")
        print("Closing program... BYE!")
        exit()

    # CREATE yaml file containing TESTING DATA:
    testINFO = {}
    testINFO["Name"] = TESTName 
    testINFO["Iterations"] = ITERATIONS
    testINFO["Objects"] = OBJECTS
    testINFO["Camera"] = CAMERA

    yamlNAME = PATH + "/RESULTS.yaml"
    with open(yamlNAME, 'w') as F:
        yaml.dump(testINFO, F)

    # Init -> ROS 2:
    rclpy.init(args=args)
    ENTITY_CLASS = EntityClient()
    OBJ = OBJECT(OBJECTS)

    # LOOP:
    i = 0
    while(i < ITERATIONS):

        i = i+1

        print("Iteration N:" + str(i))

        # 1. SPAWN OBJECTS:
        ENTITY_CLASS.spawn_REQUEST(OBJECTS)
        time.sleep(1)

        # 2. TAKE + SAVE PICTURE:
        IMG = toCV2_fromTOPIC("camera/image_raw")
        saveIMG(i, PATH, IMG)

        # 3. GET + RECORD OBJECTPOSES:
        recordObjPose(yamlNAME, i, OBJ)

        # 4. REMOVE OBJECTS:
        ENTITY_CLASS.delete_REQUEST(OBJECTS)
        time.sleep(1)

        print("")

    rclpy.shutdown()

if __name__ == '__main__':
    main()   