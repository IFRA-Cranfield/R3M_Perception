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
import os, sys, time, xacro, random

# OpenCV:
import cv2

# ROS2:
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

# Import convertIMG:
from convertIMG import toCV2_fromTOPIC

# IMPORT PythonClass -> Get OBJECT POSES:
from ObjectState import OBJECT

# IMPORT OSD and M6D:
PATH_P = os.path.join(get_package_share_directory("r3m_perception"))
PATH_OSD = PATH_P + "/oneshotdetection"
PATH_M6D = PATH_P + "/megapose6d"

sys.path.append(PATH_OSD)
from osd import OSD_CLASS

sys.path.append(PATH_M6D)
from m6d import MEGAPOSE_CLASS

# IMPORT /SpawnEntity and /DeleteEntity ROS2 Services:
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.srv import DeleteEntity

# ========================================================================================= #
# ========================================================================================= #
# Function -> Save IMAGE:
def saveIMG(i, PATH, IMG):
    imgNAME = PATH + "/" + str(i) + ".png"
    cv2.imwrite(imgNAME, IMG)

# ========================================================================================= #
# ========================================================================================= #
# R3MPerception CLASS:

class R3MP():
    
    def __init__(self):

        print("STEP 0: Initialising OSD and M6D models...")
        print("")
        
        self.OSD = OSD_CLASS()
        self.M6D = MEGAPOSE_CLASS()
        
        print("")
        print("STEP 0: Completed.")
        print("")
        
    def EXECUTE_GAZEBO(self, CAMERAType, IMGTopic, OBJList, IMGPath, i):
        
        RESULT = {}
        RESULT["Success"] = False
        
        print("========================================================")
        print("One-Shot Detection + Megapose6D Execution requested for:")
        print("  - Camera Type: " + CAMERAType)
        print("  - ROS 2 Topic of CAMERA FEED: /" + IMGTopic)
        print("  - Objects to be detected: " + str(OBJList))
        print()
        
        print("STEP 1: Getting ObjectPose from Gazebo...")
        
        # Get ObjectPose for future comparison:
        obj = OBJECT(OBJList)
        ObjectPoseList = obj.GetObjectPose()
        del obj
        
        print("STEP 1: Completed.")
        print("")
        print("STEP 2: Converting Gz CAMERA FEED to cv2 Image...")
        
        # Get IMG from ROS 2 Topic:
        inputIMG = toCV2_fromTOPIC(IMGTopic)
        saveIMG(i, IMGPath, inputIMG)
        
        print("STEP 2: Completed.")
        print("")
        print("STEP 3: Executing One-Shot Detection...")
        
        # Execute -> OSD:
        OSD_RES = self.OSD.EXECUTE_OSD(inputIMG, OBJList, CAMERAType)
        
        if OSD_RES["Success"] == False:
            print("STEP 3: Completed -> OSD Execution Failed.")
            return(RESULT)
        
        print("STEP 3: Completed. Result:")
        print("")
        
        for x in OSD_RES["Result"]:
            print("  - Object Name: "+ x["Name"])
            print("    Detection Successful? " + str(x["Success"]))
            print("    Detection Score: " + str(x["Score"]))
            print("    Bounding Box -> [tlx: "+str(x["tlx"])+", tly: "+str(x["tly"])+", brx: "+str(x["brx"])+", bry: "+str(x["bry"])+"]")
            print("")

        print("STEP 4: Executing Megapose6D...")
            
        # Execute -> M6D:
        M6D_RES = self.M6D.EXECUTE_FI(inputIMG, CAMERAType, OSD_RES)
        
        print("STEP 4: Completed. Result:")
        print("")
        
        RESULT["Result"] = []
        
        for x in M6D_RES:
            
            # 1. Prepare RESULT:
            
            RES = {}
            
            for y in ObjectPoseList:
                
                if x["Name"] == y["Name"]:
                    GzPose = {}
                    GzPose["x"] = round(y["Pose"].x, 5)
                    GzPose["y"] = round(y["Pose"].y, 5)
                    GzPose["z"] = round(y["Pose"].z, 5)
                    GzPose["qx"] = round(y["Pose"].qx, 5)
                    GzPose["qy"] = round(y["Pose"].qy, 5)
                    GzPose["qz"] = round(y["Pose"].qz, 5)
                    GzPose["qw"] = round(y["Pose"].qw, 5)
                    break
               
            M6DPose = {}
            M6DPose["x"] = x["x"]
            M6DPose["y"] = x["y"]
            M6DPose["z"] = x["z"]
            M6DPose["qx"] = x["qx"]
            M6DPose["qy"] = x["qy"]
            M6DPose["qz"] = x["qz"]
            M6DPose["qw"] = x["qw"] 
                
            RES["Name"] = x["Name"]   
            RES["GzPose"] = GzPose
            RES["M6DPose"] = M6DPose
            RESULT.append(RES)
             
            # 2. PRINT RESULT:
            print("  - Object Name: "+ x["Name"])
            print("    x -> GAZEBO: "+str(GzPose["x"])+" vs Megapose6D: "+str(M6DPose["x"]))
            print("    y -> GAZEBO: "+str(GzPose["y"])+" vs Megapose6D: "+str(M6DPose["y"]))
            print("    z -> GAZEBO: "+str(GzPose["z"])+" vs Megapose6D: "+str(M6DPose["z"]))
            print("    qx -> GAZEBO: "+str(GzPose["qx"])+" vs Megapose6D: "+str(M6DPose["qx"]))
            print("    qy -> GAZEBO: "+str(GzPose["qy"])+" vs Megapose6D: "+str(M6DPose["qy"]))
            print("    qz -> GAZEBO: "+str(GzPose["qz"])+" vs Megapose6D: "+str(M6DPose["qz"]))
            print("    qw -> GAZEBO: "+str(GzPose["qw"])+" vs Megapose6D: "+str(M6DPose["qw"]))
            print("")
            
        return(RESULT)
    
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
# ========================================================================================= #
# Function -> Create LOG FILE:
def createLOG(i, PATH):
    fileNAME = PATH + "/" + str(i) + ".txt"
    f = open(fileNAME, "x")
    f.close()
# Function -> SaveLOG:
def saveLOG(i, PATH, RES):
    
    fileNAME = PATH + "/" + str(i) + ".txt"
    f = open(fileNAME, "a")
    
    f.write("OBJECT NAME: "+ RES["Name"])
    f.write("\n")
    f.write("x -> GAZEBO: "+str(RES["GzPose"]["x"])+" vs Megapose6D: "+str(RES["M6DPose"]["x"]))
    f.write("\n")
    f.write("y -> GAZEBO: "+str(RES["GzPose"]["y"])+" vs Megapose6D: "+str(RES["M6DPose"]["y"]))
    f.write("\n")
    f.write("z -> GAZEBO: "+str(RES["GzPose"]["z"])+" vs Megapose6D: "+str(RES["M6DPose"]["z"]))
    f.write("\n")
    f.write("qx -> GAZEBO: "+str(RES["GzPose"]["qx"])+" vs Megapose6D: "+str(RES["M6DPose"]["qx"]))
    f.write("\n")
    f.write("qy -> GAZEBO: "+str(RES["GzPose"]["qy"])+" vs Megapose6D: "+str(RES["M6DPose"]["qy"]))
    f.write("\n")
    f.write("qz -> GAZEBO: "+str(RES["GzPose"]["qz"])+" vs Megapose6D: "+str(RES["M6DPose"]["qz"]))
    f.write("\n")
    f.write("qw -> GAZEBO: "+str(RES["GzPose"]["qw"])+" vs Megapose6D: "+str(RES["M6DPose"]["qw"]))
    f.write("\n")
    f.write("")
    f.write("\n")
    
    f.close()
    
# ========================================================================================= #
# ========================================= MAIN ========================================== #
# ========================================================================================= #
def main(args=None):
    
    print("")
    print("")
    
    print(" ======================== ")
    print(" R3M Perception - TESTING ")
    print(" ======================== ")
    print("")
    
    TESTName = "Test001"
    
    PATH = os.getcwd() + "/TestingResults/" + TESTName
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    else:
        print(TESTName + " folder exists. Please delete it or define a new folder name.")
        print("Closing program... BYE!")
        exit()
         
    rclpy.init(args=args)
    
    PERCEPTION_CLASS = R3MP()
    ENTITY_CLASS = EntityClient()
    
    # Define input variables:
    N = 1
    CAMERA = "lenovoFHD_gazebo"
    OBJECTS = ['adapter_plate_triangular']
    IMGTopic = "camera/image_raw"
    
    # Execute TEST:
    i = 0
    while (i < N):
        
        i = i+1
        
        print("======== "+ TESTName + " ========")
        print("Iteration N:" + str(i))
        
        # 1. SPAWN OBJECTS:
        ENTITY_CLASS.spawn_REQUEST(OBJECTS)
        
        time.sleep(1)
        
        # 2. Execute R3M Perception -> Detection:
        RESULT = PERCEPTION_CLASS.EXECUTE_GAZEBO(CAMERA, IMGTopic, OBJECTS, PATH, i)
        
        # 3. LOG RESULTS:
        createLOG(i, PATH)
        for x in RESULT["Result"]:
            saveLOG(i, PATH, x)
            
        # 4. DELETE OBJECTS:
        ENTITY_CLASS.delete_REQUEST(OBJECTS)
        
        print("")
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()