#!/usr/bin/python3
import sys
sys.dont_write_bytecode = True

# R3M Project - Cranfield University
# R3M Perception - ROS 2 Service Server (M6D)

# Import libraries:
import os, sys
import rclpy
from rclpy.node import Node

# Import -> M6D ROS 2 DATA:
from r3m_perception_data.srv import MegaPose
from objectpose_msgs.msg import ObjectPose

# IMPORT M6D CLASS:
from m6d import MEGAPOSE_CLASS

# R3M Perception PATH:
PATH_P = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception')

# Import FUNCTIONS:
PATH_F = PATH_P + "/python/functions"
sys.path.append(PATH_F)
from convertIMG import toCV2_fromROS2IMG

# Create NODE:
class serviceServer(Node):
    
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
    rclpy.init(args=None)
    r3mNode = serviceServer()
    
    r3mNode.get_logger().info("==================================================================================================")
    r3mNode.get_logger().info("[R3M Perception - M6DServer]: /R3MPerception_M6D ROS2 Service Server running, ROS2 node generated.")
    r3mNode.get_logger().info("==================================================================================================")

    # Spin SERVICE:
    rclpy.spin(r3mNode)
    
    r3mNode.destroy_node
    rclpy.shutdown()

if __name__ == '__main__':
    main()