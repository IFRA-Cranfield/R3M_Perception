import sys
sys.dont_write_bytecode = True

from osd import OSD_CLASS

from skimage import io as skimage_io
import os

CAM = "lenovoFHD_gazebo"
OSD = OSD_CLASS()

IMG_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'pictures')
IMG_PATH = IMG_PATH + "/adapter_plate_triangular.png"
IMG = skimage_io.imread(IMG_PATH)

OSD_RES = OSD.EXECUTE_OSD(IMG, ['adapter_plate_triangular'], CAM)
print("[OSD-Test]: One-Shot Detection executed! Result:")
print(OSD_RES)

exit()