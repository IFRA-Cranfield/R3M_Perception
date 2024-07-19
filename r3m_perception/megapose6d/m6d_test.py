import sys
sys.dont_write_bytecode = True

import os
from m6d import MEGAPOSE_CLASS
from skimage import io as skimage_io

CAM = "camera_ze"
BB = {'Result': [{'Name': 'bracket_planar', 'Success': True, 'Score': 0.9993587136268616, 'tlx': 373, 'tly': 36, 'brx': 714, 'bry': 269}, {'Name': 'connector_planar', 'Success': True, 'Score': 0.999691367149353, 'tlx': 329, 'tly': 357, 'brx': 482, 'bry': 657}, {'Name': 'adapter_plate_triangular', 'Success': True, 'Score': 0.9978339076042175, 'tlx': 623, 'tly': 427, 'brx': 742, 'bry': 541}], 'Success': True}

FRAME_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'pictures')
FRAME_PATH = FRAME_PATH + "/test001.png"
frame = skimage_io.imread(FRAME_PATH)

M6D = MEGAPOSE_CLASS()

RESULT = M6D.EXECUTE_FI(frame, CAM, BB)

print("")
print("========== M6D-Test (FI) - EXECUTION FINISHED ==========")
print(RESULT)

while True:

    RESULT_RTI = M6D.EXECUTE_RTI(frame)
    print("")
    print("========== M6D-Test (RTI) - EXECUTION FINISHED ==========")
    print(RESULT_RTI)