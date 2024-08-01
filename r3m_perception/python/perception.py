#!/usr/bin/python3
import sys
sys.dont_write_bytecode = True

# R3M Project - Cranfield University
# R3M Perception - ROS 2 Service Server (OSD)

# Import libraries:
import os, sys
import rclpy
from rclpy.node import Node

# Import -> OSD ROS 2 DATA:
from r3m_perception_data.srv import OneShotDet
from r3m_perception_data.msg import OneShotResult
from r3m_perception_data.srv import MegaPose
from objectpose_msgs.msg import ObjectPose

# R3M Perception PATH:
PATH_P = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception')

# IMPORT OSD CLASS:
PATH_OSD = PATH_P + "/oneshotdetection"
sys.path.append(PATH_OSD)
from osd import OSD_CLASS

# IMPORT M6D CLASS:
PATH_M6D = PATH_P + "/megapose6d"
sys.path.append(PATH_M6D)
from m6d import MEGAPOSE_CLASS

# Import FUNCTIONS:
PATH_F = PATH_P + "/python/functions"
sys.path.append(PATH_F)
from convertIMG import toCV2_fromROS2IMG

# ONE-SHOT DETECTION:
class OSDServer(Node):
    
    def __init__(self):
        
        # Initialise -> ROS 2 Service:
        super().__init__('r3m_OSDServer')
        self.srv = self.create_service(OneShotDet, "R3MPerception_OSD", self.EXECUTE_OSD)
        
        # Initialise: OSD Class:
        self.OSD = OSD_CLASS()
    
    def EXECUTE_OSD(self, request, response):
        
        # Initialise -> RESULT variable:
        RESULT = []
                
        # Convert: INPUT IMG (ROS 2 format) to OpenCV format:
        IMG_CV2 = toCV2_fromROS2IMG(request.img)
        
        # Execute -> OSD:
        OSD_RES = self.OSD.EXECUTE_OSD(IMG_CV2, request.cadlist, request.camera)
        
        if OSD_RES["Success"] == False:
            response.success = False
            return(response)
            
        # Get RESPONSE:
        for x in OSD_RES["Result"]:
            
            RES = OneShotResult()
            
            RES.cadname = x["Name"]
            RES.success = x["Success"]
            RES.score = x["Score"]
            RES.tlx = x["tlx"]
            RES.tly = x["tly"]
            RES.brx = x["brx"]
            RES.bry = x["bry"]
            
            RESULT.append(RES)
            
        response.success = True
        response.result = RESULT
        
        return(response)
    
# MEGAPOSE6D:
class M6DServer(Node):
    
    def __init__(self):
        
        # Initialise -> ROS 2 Service:
        super().__init__('r3m_M6DServer')
        self.srv = self.create_service(MegaPose, "R3MPerception_M6D", self.EXECUTE_M6D)
        
        # Initialise: M6D Class:
        self.M6D = MEGAPOSE_CLASS()
    
    def EXECUTE_M6D(self, request, response):
        
        # Initialise -> RESULT variable:
        RESULT = []
                
        # Convert: INPUT IMG (ROS 2 format) to OpenCV format:
        IMG_CV2 = toCV2_fromROS2IMG(request.img)
        
        # CONVERT INPUT:
        INPUT = {}
        INPUT["Success"] = True
        INPUT["Result"] = []
        
        for x in request.input:
            
            INP = {}
            INP["Name"] = x.cadname
            INP["Success"] = x.success
            INP["Score"] = x.score
            INP["tlx"] = x.tlx
            INP["tly"] = x.tly
            INP["brx"] = x.brx
            INP["bry"] = x.bry
            INPUT["Result"].append(INP)
        
        # Execute -> M6D:
        M6D_RES = self.M6D.EXECUTE_FI(IMG_CV2, request.camera, INPUT)
            
        # Get RESPONSE:
        for x in M6D_RES:
            
            RES = ObjectPose()
            
            RES.objectname = x["Name"]
            RES.x = x["x"]
            RES.y = x["y"]
            RES.z = x["z"]
            RES.qx = x["qx"]
            RES.qy = x["qy"]
            RES.qz = x["qz"]
            RES.qw = x["qw"] 
            
            RESULT.append(RES)
            
        response.success = True
        response.result = RESULT
        
        return(response)

# =================== MAIN =================== #
def main(args=None):
    
    # Initialise NODE:
    rclpy.init(args=args)
    OSD_NODE = OSDServer()
    M6D_NODE = M6DServer()
    
    OSD_NODE.get_logger().info("==================================================================================================")
    OSD_NODE.get_logger().info("[R3M Perception - SERVICE SERVERS]: /R3MPerception_OSD and /R3MPerception_M6D ROS2 Services ready.")
    OSD_NODE.get_logger().info("==================================================================================================")

    # Spin SERVICES:
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(OSD_NODE)
    executor.add_node(M6D_NODE)
    executor.spin()
    
    OSD_NODE.destroy_node()
    M6D_NODE.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()