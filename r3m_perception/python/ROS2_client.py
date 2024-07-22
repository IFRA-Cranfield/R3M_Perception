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
from r3m_perception_data.srv import MegaPose
from r3m_perception_data.srv import OneShotDet

# IMPORT -> IMG conversion functions:
PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'python')
sys.path.append(PATH)
from convertIMG import toROS2IMG_fromTOPIC, toROS2IMG_fromCV2

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
        
# CLASS -> EXECUTE PERCEPTION:
class R3MPerception():
    
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

# =================== MAIN =================== #
def main(args=None):
   
    # Initialise NODE:
    rclpy.init(args=args)
    r3mPERCEPTION = R3MPerception()
    
    print("=====================================================================================================")
    print("[R3M Perception - ROS2 Client]: Connected to /R3MPerception_OSD and /R3MPerception_M6D ROS2 Services.")
    print("=====================================================================================================")

    # 1. DEFINE INPUT VARIABLES to IMG:
    OBJECTS = ['adapter_plate_triangular']
    
    CAMERA = "camera_ze"
    #CAMERA = "lenovoFHD_gazebo"
    
    #IMGTopic = "camera/image_raw"
    #IMG = toROS2IMG_fromTOPIC(IMGTopic)
    img_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'pictures')
    img_PATH = img_PATH + "/test001.png"
    img = skimage_io.imread(img_PATH)
    IMG = toROS2IMG_fromCV2(img)
    
    # 2. EXECUTE OSD:
    OSD_RES = r3mPERCEPTION.EXECUTE_OSD(IMG, CAMERA, OBJECTS)
    
    # 3. EXECUTE M6D:
    M6D_RES = r3mPERCEPTION.EXECUTE_M6D(IMG, CAMERA, OSD_RES.result)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()