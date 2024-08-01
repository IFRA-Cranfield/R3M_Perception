#!/usr/bin/python3
import sys
sys.dont_write_bytecode = True

# R3M Project - Cranfield University
# R3M Perception - ROS 2 Service Client (OSD)

# Import libraries:
import os, sys
import rclpy
from rclpy.node import Node
from skimage import io as skimage_io

# Import -> OSD ROS 2 DATA:
from r3m_perception_data.srv import OneShotDet

# R3M Perception PATH:
PATH_P = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception')

# Import FUNCTIONS:
PATH_F = PATH_P + "/python/functions"
sys.path.append(PATH_F)
from convertIMG import toROS2IMG_fromTOPIC, toROS2IMG_fromCV2

# Create NODE:
class serviceClient(Node):
    
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

# =================== MAIN =================== #
def main(args=None):
   
    # Initialise NODE:
    rclpy.init(args=args)
    r3mNode = serviceClient()
    
    r3mNode.get_logger().info("=======================================================================================================")
    r3mNode.get_logger().info("[R3M Perception - OSDClient]: Connected to /R3MPerception_OSD ROS2 Service Server, ROS2 node generated.")
    r3mNode.get_logger().info("=======================================================================================================")

    # 1. DEFINE INPUT VARIABLES to IMG:
    OBJECTS = ['adapter_plate_triangular']
    CAMERA = "camera_ze"
    
    IMG_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'pictures')
    IMG_PATH = IMG_PATH + "/test001.png"
    IMG_CV2 = skimage_io.imread(IMG_PATH)
    IMG = toROS2IMG_fromCV2(IMG_CV2)
    
    # If willing to test w/ an image from Gz simulation:
    # CAMERA = "lenovoFHD_gazebo"
    # IMGTopic = "camera/image_raw"
    # IMG = toROS2IMG_fromTOPIC(IMGTopic)
    
    # 2. EXECUTE SERVICE:
    r3mNode.EXECUTE_OSD(IMG, OBJECTS, CAMERA)

    while rclpy.ok():
        rclpy.spin_once(r3mNode)
        
        if r3mNode.future.done():
            try:
                response = r3mNode.future.result()
            except Exception as exc:
                r3mNode.get_logger().info("[R3M Perception - OSDClient]: Service call failed -> " + str(exc))
            else:
                
                r3mNode.get_logger().info("[R3M Perception - OSDClient]: ROS 2 Service Call executed. RESULTS:")
                
                # LOG RESULTS:
                if response.success == False:
                    r3mNode.get_logger().info("[R3M Perception - OSDClient]: OSD Execution Failed.")
                else:
                    for x in response.result:
                        r3mNode.get_logger().info("  - Object Name: "+ x.cadname)
                        r3mNode.get_logger().info("    Detection Successful? " + str(x.success))
                        r3mNode.get_logger().info("    Detection Score: " + str(x.score))
                        r3mNode.get_logger().info("    Bounding Box -> [tlx: "+str(x.tlx)+", tly: "+str(x.tly)+", brx: "+str(x.brx)+", bry: "+str(x.bry)+"]")
                
            break
    
    r3mNode.destroy_node
    rclpy.shutdown()

if __name__ == '__main__':
    main()