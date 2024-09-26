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
import os, sys, time, xacro, random, yaml, threading

# OpenCV:
import cv2

# ROS2:
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

# IMPORT /SpawnEntity and /DeleteEntity ROS2 Services:
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.srv import DeleteEntity
from std_srvs.srv import Empty

import math

# R3M Perception PATH:
PATH_P = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception')

# Import FUNCTIONS:
PATH_F = PATH_P + "/python/functions"
sys.path.append(PATH_F)
# IMPORT PythonClass -> Get OBJECT POSES:
from ObjectState import OBJECT
# Import convertIMG:
from convertIMG import imgSUB

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
def recordObjPose(yamlNAME, i, ObjectPoseList):

    with open(yamlNAME, 'r') as F:
        testINFO = yaml.safe_load(F)

    inputDICT = {}

    for x in ObjectPoseList:
        
        POSE = {}
        POSE["x"] = round(x["Pose"].x, 5)
        POSE["y"] = round(x["Pose"].y, 5)
        POSE["z"] = round(x["Pose"].z, 5)
        POSE["qx"] = round(x["Pose"].qx, 5)
        POSE["qy"] = round(x["Pose"].qy, 5)
        POSE["qz"] = round(x["Pose"].qz, 5)
        POSE["qw"] = round(x["Pose"].qw, 5)

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
        self.cli_PAUSE = self.create_client(Empty, "/pause_physics")
        self.cli_UNPAUSE = self.create_client(Empty, "/unpause_physics")

        # Declare REQUEST variable (of CUSTOM DATA type):
        self.req_SPAWN = SpawnEntity.Request()  
        self.req_DELETE = DeleteEntity.Request()
        self.req_EMPTY = Empty.Request()

    def euler_to_quaternion(self, yaw, pitch, roll):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return w, x, y, z

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
            self.req_SPAWN.initial_pose.position.z = 0.95
            self.yaw = random.uniform(0.0,6.28)
            self.req_SPAWN.initial_pose.orientation.w, self.req_SPAWN.initial_pose.orientation.x, self.req_SPAWN.initial_pose.orientation.y, self.req_SPAWN.initial_pose.orientation.z = self.euler_to_quaternion(self.yaw, 0, 0)
            # Add here -> Random orientation.

            # Assign RESULT value (future):
            self.future_SPAWN = self.cli_SPAWN.call_async(self.req_SPAWN)

    def delete_REQUEST(self, ObjectList):

        for x in ObjectList:
            self.req_DELETE.name = x
            self.future_DELETE = self.cli_DELETE.call_async(self.req_DELETE)

    def pause_REQUEST(self):
        self.future_PAUSE = self.cli_PAUSE.call_async(self.req_EMPTY)

    def unpause_REQUEST(self):
        self.future_UNPAUSE = self.cli_UNPAUSE.call_async(self.req_EMPTY)

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
    TESTName = "Test002"
    ITERATIONS = 10
    OBJECTS = ['adapter_plate_square']
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

    # Python THREAD for -> IMG SUBSCRIBER, OBJECTPOSE SUBSCRIBER:
    subIMG = imgSUB("camera/image_raw")
    subOBJECTPOSE = OBJECT(OBJECTS)
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(subIMG)
    executor.add_node(subOBJECTPOSE)
    THREAD = threading.Thread(target=executor.spin, daemon=True)
    THREAD.start()

    # LOOP:
    i = 0
    while(i < ITERATIONS):

        i = i+1

        print("Iteration N:" + str(i))

        # 1. SPAWN OBJECTS:
        ENTITY_CLASS.spawn_REQUEST(OBJECTS)
        time.sleep(int(len(OBJECTS)))
        ENTITY_CLASS.pause_REQUEST()
        
        # 2. TAKE + SAVE PICTURE:
        IMG = subIMG.toCV2_fromTOPIC()
        saveIMG(i, PATH, IMG)

        # 3. GET + RECORD OBJECTPOSES:
        ObjectList = subOBJECTPOSE.GetObjectPose_LAST()
        recordObjPose(yamlNAME, i, ObjectList)

        # 4. REMOVE OBJECTS:
        ENTITY_CLASS.unpause_REQUEST()
        ENTITY_CLASS.delete_REQUEST(OBJECTS)
        time.sleep(int(len(OBJECTS)))

        print("")

    rclpy.shutdown()

if __name__ == '__main__':
    main()   