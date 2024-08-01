#!/usr/bin/python3
import sys
sys.dont_write_bytecode = True

# R3M Project - Cranfield University
# R3M Perception - ROS 2 Service Client (M6D)

# Import libraries:
import os, sys
import rclpy
from rclpy.node import Node
from skimage import io as skimage_io

# Import -> M6D ROS 2 DATA:
from r3m_perception_data.srv import MegaPose
from r3m_perception_data.msg import OneShotResult

# R3M Perception PATH:
PATH_P = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception')

# Import FUNCTIONS:
PATH_F = PATH_P + "/python/functions"
sys.path.append(PATH_F)
from convertIMG import toROS2IMG_fromTOPIC, toROS2IMG_fromCV2

# Create NODE:
class serviceClient(Node):
    
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

# =================== MAIN =================== #
def main(args=None):
   
    # Initialise NODE:
    rclpy.init(args=args)
    r3mNode = serviceClient()
    
    r3mNode.get_logger().info("=======================================================================================================")
    r3mNode.get_logger().info("[R3M Perception - M6DClient]: Connected to /R3MPerception_M6D ROS2 Service Server, ROS2 node generated.")
    r3mNode.get_logger().info("=======================================================================================================")

    # 1. DEFINE INPUT VARIABLES to IMG:
    CAMERA = "camera_ze"
    
    IMG_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'pictures')
    IMG_PATH = IMG_PATH + "/test001.png"
    IMG_CV2 = skimage_io.imread(IMG_PATH)
    IMG = toROS2IMG_fromCV2(IMG_CV2)
    
    # M6D OUTPUT:
    INPUT_M6D = []
    
    INPUT = OneShotResult()
    INPUT.cadname = 'adapter_plate_triangular'
    INPUT.success = True
    INPUT.score = 0.98
    INPUT.tlx = 623
    INPUT.tly = 427
    INPUT.brx = 742
    INPUT.bry = 541 # Results manually obtained from OSD_Client execution.
    INPUT_M6D.append(INPUT)
    
    # If willing to test w/ an image from Gz simulation:
    # CAMERA = "lenovoFHD_gazebo"
    # IMGTopic = "camera/image_raw"
    # IMG = toROS2IMG_fromTOPIC(IMGTopic)
    
    # 2. EXECUTE SERVICE:
    r3mNode.EXECUTE_M6D(IMG, CAMERA, INPUT_M6D)

    while rclpy.ok():
        rclpy.spin_once(r3mNode)
        
        if r3mNode.future.done():
            try:
                response = r3mNode.future.result()
            except Exception as exc:
                r3mNode.get_logger().info("[R3M Perception - M6DClient]: Service call failed -> " + str(exc))
            else:
                
                r3mNode.get_logger().info("[R3M Perception - M6DClient]: ROS 2 Service Call executed. RESULTS:")
            
                for x in response.result:
                    r3mNode.get_logger().info("  - Object Name: "+ x.objectname)
                    r3mNode.get_logger().info("    x -> " + str(x.x))
                    r3mNode.get_logger().info("    y -> " + str(x.y))
                    r3mNode.get_logger().info("    z -> " + str(x.z))
                    r3mNode.get_logger().info("    qx -> " + str(x.qx))
                    r3mNode.get_logger().info("    qy -> " + str(x.qy))
                    r3mNode.get_logger().info("    qz -> " + str(x.qz))
                    r3mNode.get_logger().info("    qw -> " + str(x.qw))
            break
    
    r3mNode.destroy_node
    rclpy.shutdown()

if __name__ == '__main__':
    main()