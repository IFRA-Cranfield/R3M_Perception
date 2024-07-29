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

# Import -> OSD ROS 2 DATA:
from r3m_perception_data.srv import MegaPose
from r3m_perception_data.srv import OneShotDet

# R3M Perception PATH:
PATH_P = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception')

# Import FUNCTIONS:
PATH_F = PATH_P + "/python/functions"
sys.path.append(PATH_F)
# Import convertIMG:
from convertIMG import toROS2IMG_fromCV2

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
# R3MP CLASS:

class R3MP():
    
    def __init__(self, CAMERAType, ObjectList, PATH):

        self.CAMERA = CAMERAType
        self.OBJECTS = ObjectList
        self.PATH = PATH
        self.yamlNAME = PATH + "/RESULTS.yaml"

        print("STEP 0: Initialising OSD and M6D models...")
        print("")
        
        self.Perception = R3MPerceptionClient()
        
        print("")
        print("STEP 0: Completed.")
        print("")

    def EXECUTE_OSDM6D(self, i):

        RESULT = {}
        RESULT["Success"] = False

        imgPATH = self.PATH + "/" + str(i) + ".png"

        print("========================================================")
        print("One-Shot Detection + Megapose6D Execution requested for:")
        print("  - Camera Type: " + self.CAMERAType)
        print("  - IMAGE: " + imgPATH)
        print("  - Objects to be detected: " + str(self.OBJECTS))
        print()
        
        print("STEP 1: Getting ObjectPose from Gazebo...")

        # Convert cv2 image to ROS 2 image:
        IMG_cv2 = cv2.imread(imgPATH)
        IMG_ROS2 = toROS2IMG_fromCV2(IMG_cv2)

        print("STEP 1: Completed.")
        print("")
        print("STEP 2: Executing One-Shot Detection...")
        
        # Execute -> OSD:
        OSD_RES = self.Perception.EXECUTE_OSD(IMG_ROS2, self.CAMERAType, self.OBJECTS)
        
        if OSD_RES.success == False:
            print("STEP 2: Completed -> OSD Execution Failed.")
            return(RESULT)
        
        print("STEP 2: Completed. Result:")
        print("")
        
        for x in OSD_RES.result:
            print("  - Object Name: "+ x.cadname)
            print("    Detection Successful? " + str(x.success))
            print("    Detection Score: " + str(x.score))
            print("    Bounding Box -> [tlx: "+str(x.tlx)+", tly: "+str(x.tly)+", brx: "+str(x.brx)+", bry: "+str(x.bry)+"]")
            print("")

        print("STEP 3: Executing Megapose6D...")
            
        # Execute -> M6D:
        M6D_RES = self.Perception.EXECUTE_M6D(IMG_ROS2, self.CAMERAType, OSD_RES.result)
        
        print("STEP 3: Completed. Result:")
        print("")

        RESULT["Result"] = []

        with open(self.yamlNAME, 'r') as F:
            testINFO = yaml.safe_load(F)
        
        for x in M6D_RES.result:

            POSE = {}
            POSE["x"] = x.x
            POSE["y"] = x.y
            POSE["z"] = x.z
            POSE["qx"] = x.qx
            POSE["qy"] = x.qy
            POSE["qz"] = x.qz
            POSE["qw"] = x.qw

            testINFO[str(i)][x.objectname]["Perception"] = POSE

            print("R3M Perception estimation  of object -> " + x["Name"] + " for image -> " + imgPATH + " recorded.")
    
# ========================================================================================= #
# ========================================= MAIN ========================================== #
# ========================================================================================= #
def main(args=None):
    
    print("")
    print("")
    
    print(" ======================== ")
    print(" R3M Perception - TESTING ")
    print(" Execute OSD+M6D from IMG ")
    print(" ======================== ")
    print("")

    # INPUT VARIABLES:
    FOLDER = os.path.join(os.path.expanduser('~'), 'PerceptionTesting')
    TESTName = "Test001"
    
    PATH = FOLDER + "/" + TESTName
    if not os.path.exists(PATH):
        print(TESTName + " folder does not exist. Please change the FOLDER NAME.")
        print("Closing program... BYE!")
        exit()
        
    yamlNAME = PATH + "/RESULTS.yaml"
    
    with open(yamlNAME, 'r') as F:
        testINFO = yaml.safe_load(F)

    ITERATIONS = testINFO["Iterations"]
    ObjectList = testINFO["Objects"]
    CAMERA = testINFO["Camera"]

    # Init -> ROS 2:
    rclpy.init(args=args)

    PERCEPTIONCLIENT = R3MP(CAMERA, ObjectList, PATH)

    # LOOP:
    i = 0
    while(i < ITERATIONS):

        i = i+1

        print("Iteration N:" + str(i))
        print("")

        PERCEPTIONCLIENT.EXECUTE_OSDM6D(i)

        print("")

    rclpy.shutdown()

if __name__ == '__main__':
    main()   