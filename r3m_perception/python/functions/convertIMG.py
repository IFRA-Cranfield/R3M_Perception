#!/usr/bin/python3
import sys
sys.dont_write_bytecode = True

# IMPORT:
import time

# IMPORT OpenCV:
from cv_bridge import CvBridge, CvBridgeError

# Import ROS 2:
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# =========================================== #
# Image SUBSCRIBER class:
class imgSUB(Node):
    
    def __init__(self, TOPICNAME):
        
        # Declare NODE:
        super().__init__("r3m_imgSUB")
        
        # Declare SUBSCRIBER:
        self.subscription = self.create_subscription(
            Image,                                             
            TOPICNAME, 
            self.listener_callback, 
            10) 
        self.subscription 
        
        self.IMAGE = Image()

    def listener_callback(self, IMG):
        self.IMAGE = IMG

# =========================================== #
# FUNCTION -> GET image (CV2 format) from ROS 2 topic:
def toCV2_fromTOPIC(TOPIC):
    
    # INITIALISE CLASSES:
    BRIDGE = CvBridge()
    SUB = imgSUB(TOPIC)
    
    # Get IMAGE (sensor_msgs/Image format) from ROS 2 TOPIC:
    T = time.time() + 0.25
    while time.time() < T:
        rclpy.spin_once(SUB)   
    IMG_ROS2 = SUB.IMAGE
    
    # CONVERT:
    IMG_CV2 = BRIDGE.imgmsg_to_cv2(IMG_ROS2, "passthrough")
    
    # Delete CLASS INSTANCES:
    del BRIDGE, SUB
    
    # Return IMAGE (OpenCV format):
    return(IMG_CV2)

# =========================================== #
# FUNCTION -> GET image (ROS 2 format) from ROS 2 topic:
def toROS2IMG_fromTOPIC(TOPIC):
    
    # INITIALISE CLASSES:
    SUB = imgSUB(TOPIC)
    
    # Get IMAGE (sensor_msgs/Image format) from ROS 2 TOPIC:
    T = time.time() + 0.25
    while time.time() < T:
        rclpy.spin_once(SUB)   
    IMG_ROS2 = SUB.IMAGE
    
    # Delete CLASS INSTANCES:
    del SUB
    
    # Return IMAGE (OpenCV format):
    return(IMG_ROS2)

# =========================================== #
# FUNCTION -> GET image from ROS 2 IMG:
def toCV2_fromROS2IMG(IMG_ROS2):
    
    # INITIALISE CLASSES:
    BRIDGE = CvBridge()
    
    # CONVERT:
    IMG_CV2 = BRIDGE.imgmsg_to_cv2(IMG_ROS2, "passthrough")
    
    # Delete CLASS INSTANCES:
    del BRIDGE
    
    # Return IMAGE (OpenCV format):
    return(IMG_CV2)

# =========================================== #
# FUNCTION -> GET image from ROS 2 IMG:
def toROS2IMG_fromCV2(IMG_CV2):
    
    # INITIALISE CLASSES:
    BRIDGE = CvBridge()
    
    # CONVERT:
    IMG_ROS2 = BRIDGE.cv2_to_imgmsg(IMG_CV2, "passthrough")
    
    # Delete CLASS INSTANCES:
    del BRIDGE
    
    # Return IMAGE (OpenCV format):
    return(IMG_ROS2)