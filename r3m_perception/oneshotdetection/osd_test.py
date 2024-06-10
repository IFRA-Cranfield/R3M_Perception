import sys
sys.dont_write_bytecode = True

from osd import OSD_CLASS

from skimage import io as skimage_io
import os

CAD = "adapter_plate_square"
CAM = "lenovoFHD_gazebo"
OSD = OSD_CLASS(CAD, CAM)

IMG_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'pictures')
IMG_PATH = IMG_PATH + "/" + CAD + ".png"
IMG = skimage_io.imread(IMG_PATH)
OSD.EXECUTE_OSD(IMG)

exit()