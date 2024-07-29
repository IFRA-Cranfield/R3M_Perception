#!/usr/bin/python3

# # # # # # # # # # # # # # # # # #                                  
#                                 #
#   ===== COPYRIGHT HERE =====    #
#                                 #
# # # # # # # # # # # # # # # # # #

# ========================================================================================= #
# ======================================== INCLUDE ======================================== #
# ========================================================================================= #

# System:
import time

# ROS2:
import rclpy
from rclpy.node import Node

# CUSTOM ROS2 MSG/SRV/ACTION:
from objectpose_msgs.msg import ObjectPose

# ========================================================================================= #
# =================================== CLASSES/FUNCTIONS =================================== #
# ========================================================================================= #

# ========================================================================================= #
# OBJECT CLASS:
class OBJECT(Node):

    def __init__(self, ObjectList):
              
        # "ObjectList": ["", "", ""]

        super().__init__("R3MPerception_ObjectState")
        
        self.SUBList = []
        self.ObjectPoseList = []
        
        for x in ObjectList:
            
            TopicName = "/" + x + "/ObjectPose"
            self.SUBList.append(self.create_subscription(ObjectPose, TopicName, self.CALLBACK_FN, 1))
            
            OBJ = {}
            OBJ["Name"] = x
            OBJ["Pose"] = None
            self.ObjectPoseList.append(OBJ) 
        
    def CALLBACK_FN(self, OBJ):
        
        for x in self.ObjectPoseList:
            if (OBJ.objectname == x["Name"]):
                x["Pose"] = OBJ

    def GetObjectPose(self):
        
        # 1. Spin node:
        T = time.time() + 0.25
        while time.time() < T:
            rclpy.spin_once(self)

        # 2. RETURN:
        return(self.ObjectPoseList)
    
    def GetObjectPose_LAST(self):
        return(self.ObjectPoseList)