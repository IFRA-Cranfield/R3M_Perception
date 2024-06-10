import sys
sys.dont_write_bytecode = True

import os
from m6d import MEGAPOSE_CLASS
from skimage import io as skimage_io

CAD = "adapter_plate_square"
CAM = "lenovoFHD_gazebo"
M6D = MEGAPOSE_CLASS(CAD,CAM)

BB = {}
BB['tlx']=343
BB['tly']=474
BB['brx']=426
BB['bry']=554

FRAME_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'pictures')
FRAME_PATH = FRAME_PATH + "/" + CAD + ".png"
frame = skimage_io.imread(FRAME_PATH)

RESULT = M6D.EXECUTE_FI(frame, BB)
print("")
print("========== M6D (FI) - EXECUTION FINISHED ==========")
print(RESULT)

while True:

    RESULT_RTI = M6D.EXECUTE_RTI(frame)
    print("")
    print("========== M6D (RTI) - EXECUTION FINISHED ==========")
    print(RESULT_RTI)