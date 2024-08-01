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

# Import -> OSD ROS 2 DATA:
from r3m_perception_data.srv import MegaPose
from r3m_perception_data.srv import OneShotDet

# IMPORT /SpawnEntity and /DeleteEntity ROS2 Services:
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.srv import DeleteEntity

# R3M Perception PATH:
PATH_P = os.path.join(get_package_share_directory("r3m_perception"))

# Import FUNCTIONS:
PATH_F = PATH_P + "/python/functions"
sys.path.append(PATH_F)
# Import convertIMG:
from convertIMG import toROS2IMG_fromTOPIC, toCV2_fromROS2IMG
# IMPORT PythonClass -> Get OBJECT POSES:
from ObjectState import OBJECT

# ========================================================================================= #
# ========================================================================================= #
# Function -> Save IMAGE:
def saveIMG(i, PATH, IMG):
    imgNAME = PATH + "/" + str(i) + ".png"
    cv2.imwrite(imgNAME, IMG)

# ========================================================================================= #
# ========================================================================================= #
# CLASSES -> OSD and M6D ROS 2 Service Clients:

# OSD NODE:
class OSDClient(Node):
    
    def __init__(self):
        
        super().__init__('r3m_OSDClient')
        self.cli = self.create_client(OneShotDet, "R3MPerception_OSD")
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("[R3M Perception - OSDClient]: /R3MPerception_OSD ROS Service not available, waiting...")
        self.req = OneShotDet.Request()
        self.get_logger().info("[R3M Perception - OSDClient]: /R3MPerception_OSD ROS2.0 SERVICE detected!")

    def EXECUTE_OSD(self, IMG, CADList, CAM):
        
        self.get_logger().info("[R3M Perception - OSDClient]: Executing OSD Request...")

        self.req.cadlist = CADList
        self.req.camera = CAM
        self.req.img = IMG
        
        self.future = self.cli.call_async(self.req)    
         
# M6D NODE:
class M6DClient(Node):
    
    def __init__(self):
        
        super().__init__('r3m_M6DClient')
        self.cli = self.create_client(MegaPose, "R3MPerception_M6D")
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("[R3M Perception - M6DClient]: /R3MPerception_M6D ROS Service not available, waiting...")
        self.req = MegaPose.Request()
        self.get_logger().info("[R3M Perception - M6DClient]: /R3MPerception_M6D ROS2.0 SERVICE detected!")

    def EXECUTE_M6D(self, IMG, CAM, INPUT):
        
        self.get_logger().info("[R3M Perception - M6DClient]: Executing M6D Request...")
        
        self.req.input = INPUT
        self.req.camera = CAM
        self.req.img = IMG
        
        self.future = self.cli.call_async(self.req)
        
# CLASS -> R3MPerceptionClient:
class R3MPerceptionClient():
    
    def __init__(self):
        
        self.OSDNode = OSDClient()
        self.M6DNode = M6DClient()
        
    def EXECUTE_OSD(self, IMG, CAMERA, OBJECTS):
        
        self.OSDNode.EXECUTE_OSD(IMG, OBJECTS, CAMERA)

        while rclpy.ok():
            rclpy.spin_once(self.OSDNode)
            
            if self.OSDNode.future.done():
                try:
                    response = self.OSDNode.future.result()
                except Exception as exc:
                    self.OSDNode.get_logger().info("[R3M Perception - OSDClient]: Service call failed -> " + str(exc))
                else:
                    
                    self.OSDNode.get_logger().info("[R3M Perception - OSDClient]: ROS 2 Service Call executed. RESULTS:")
                    
                    # LOG RESULTS:
                    if response.success == False:
                        self.OSDNode.get_logger().info("[R3M Perception - OSDClient]: OSD Execution Failed.")
                    else:
                        for x in response.result:
                            self.OSDNode.get_logger().info("  - Object Name: "+ x.cadname)
                            self.OSDNode.get_logger().info("    Detection Successful? " + str(x.success))
                            self.OSDNode.get_logger().info("    Detection Score: " + str(x.score))
                            self.OSDNode.get_logger().info("    Bounding Box -> [tlx: "+str(x.tlx)+", tly: "+str(x.tly)+", brx: "+str(x.brx)+", bry: "+str(x.bry)+"]")
                    
                break
            
        return(response)
    
    def EXECUTE_M6D(self, IMG, CAMERA, INPUT_M6D):
        
        self.M6DNode.EXECUTE_M6D(IMG, CAMERA, INPUT_M6D)

        while rclpy.ok():
            rclpy.spin_once(self.M6DNode)
            
            if self.M6DNode.future.done():
                try:
                    response = self.M6DNode.future.result()
                except Exception as exc:
                    self.M6DNode.get_logger().info("[R3M Perception - M6DClient]: Service call failed -> " + str(exc))
                else:
                    
                    self.M6DNode.get_logger().info("[R3M Perception - M6DClient]: ROS 2 Service Call executed. RESULTS:")
                
                    for x in response.result:
                        self.M6DNode.get_logger().info("  - Object Name: "+ x.objectname)
                        self.M6DNode.get_logger().info("    x -> " + str(x.x))
                        self.M6DNode.get_logger().info("    y -> " + str(x.y))
                        self.M6DNode.get_logger().info("    z -> " + str(x.z))
                        self.M6DNode.get_logger().info("    qx -> " + str(x.qx))
                        self.M6DNode.get_logger().info("    qy -> " + str(x.qy))
                        self.M6DNode.get_logger().info("    qz -> " + str(x.qz))
                        self.M6DNode.get_logger().info("    qw -> " + str(x.qw))
                break
            
        return(response)

# ========================================================================================= #
# ========================================================================================= #
# R3MPerception CLASS:

class R3MP():
    
    def __init__(self):

        print("STEP 0: Initialising OSD and M6D models...")
        print("")
        
        self.Perception = R3MPerceptionClient()
        
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
        inputIMG = toROS2IMG_fromTOPIC(IMGTopic)
        outputIMG = toCV2_fromROS2IMG(inputIMG)
        saveIMG(i, IMGPath, outputIMG)
        
        print("STEP 2: Completed.")
        print("")
        print("STEP 3: Executing One-Shot Detection...")
        
        # Execute -> OSD:
        OSD_RES = self.Perception.EXECUTE_OSD(inputIMG, CAMERAType, OBJList)
        
        if OSD_RES.success == False:
            print("STEP 3: Completed -> OSD Execution Failed.")
            return(RESULT)
        
        print("STEP 3: Completed. Result:")
        print("")
        
        for x in OSD_RES.result:
            print("  - Object Name: "+ x.cadname)
            print("    Detection Successful? " + str(x.success))
            print("    Detection Score: " + str(x.score))
            print("    Bounding Box -> [tlx: "+str(x.tlx)+", tly: "+str(x.tly)+", brx: "+str(x.brx)+", bry: "+str(x.bry)+"]")
            print("")

        print("STEP 4: Executing Megapose6D...")
            
        # Execute -> M6D:
        M6D_RES = self.Perception.EXECUTE_M6D(inputIMG, CAMERAType, OSD_RES.result)
        
        print("STEP 4: Completed. Result:")
        print("")
        
        RESULT["Result"] = []
        
        for x in M6D_RES.result:
            
            # 1. Prepare RESULT:
            
            RES = {}
            
            for y in ObjectPoseList:
                
                if x.objectname == y["Name"]:
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
            M6DPose["x"] = x.x
            M6DPose["y"] = x.y
            M6DPose["z"] = x.z
            M6DPose["qx"] = x.qx
            M6DPose["qy"] = x.qy
            M6DPose["qz"] = x.qz
            M6DPose["qw"] = x.qw
                
            RES["Name"] = x.objectname
            RES["GzPose"] = GzPose
            RES["M6DPose"] = M6DPose
            RESULT["Result"].append(RES)
             
            # 2. PRINT RESULT:
            print("  - Object Name: "+ x.objectname)
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
    N = 50
    CAMERA = "lenovoFHD_gazebo"
    OBJECTS = ['adapter_plate_triangular']
    IMGTopic = "camera/image_raw"
    
    # Execute TEST:
    i = 0
    while (i < N):
        
        i = i+1
        
        print("======== "+ TESTName + " ========")
        print("")
        print("Iteration N:" + str(i))
        
        # 1. SPAWN OBJECTS:
        #ENTITY_CLASS.spawn_REQUEST(OBJECTS)
        
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