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

# IMPORT OSD CLASS:
from osd import OSD_CLASS

# R3M Perception PATH:
PATH_P = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception')

# Import FUNCTIONS:
PATH_F = PATH_P + "/python/functions"
sys.path.append(PATH_F)
from convertIMG import toCV2_fromROS2IMG

# Create NODE:
class serviceServer(Node):
    
    def __init__(self):
        
        # Initialise: OSD Class:
        self.OSD = OSD_CLASS()
        
        # Initialise -> ROS 2 Service:
        super().__init__('r3m_OSDServer')
        self.srv = self.create_service(OneShotDet, "R3MPerception_OSD", self.EXECUTE_OSD)
    
    def EXECUTE_OSD(self, request, response):
        
        # Initialise -> RESULT variable:
        RESULT = []
                
        # Convert: INPUT IMG (ROS 2 format) to OpenCV format:
        IMG_CV2 = toCV2_fromROS2IMG(request.img)
        
        # Execute -> OSD:
        OSD_RES = self.OSD.EXECUTE_OSD(IMG_CV2, request.cadlist, request.camera, request.idx)
        
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

# =================== MAIN =================== #
def main(args=None):
    
    # Initialise NODE:
    rclpy.init(args=args)
    r3mNode = serviceServer()
    
    print("")
    
    print("==================================================================================================")
    print("[R3M Perception - OSDServer]: /R3MPerception_OSD ROS2 Service Server running, ROS2 node generated.")
    print("==================================================================================================")

    print("")

    # Spin SERVICE:
    rclpy.spin(r3mNode)
    
    r3mNode.destroy_node
    rclpy.shutdown()

if __name__ == '__main__':
    main()