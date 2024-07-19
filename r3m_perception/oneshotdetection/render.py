import os
import open3d as o3d
import bpy
import sys

if len(sys.argv)==2:
    CADName = sys.argv[1]
else:
    exit()

# PATHS TO IMG and CAD:
cad_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'cad', 'mesh', CADName)
img_PATH = os.path.join(os.path.expanduser('~'), 'dev_ws', 'src', 'R3M_Perception', 'r3m_perception', 'cad', 'osd_execution')

cadPATH = cad_PATH + "/" + CADName + ".ply"
objPATH = img_PATH + "/" + CADName + ".obj"
imgPATH = img_PATH + "/OSD_Inputs/OSD_Input_" + CADName + ".png"

# ========================================= #
# 1. CONVERT from ply to obj:

# Load the PLY file:
mesh = o3d.io.read_triangle_mesh(cadPATH)

# Check if the mesh is empty:
if mesh.is_empty():
    exit()

# Save the mesh as an OBJ file:
o3d.io.write_triangle_mesh(objPATH, mesh)

# ========================================= #
# 2. RENDER:

# Clear existing objects in the scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import the OBJ file
bpy.ops.import_scene.obj(filepath=objPATH)

# Set up rendering settings
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'  # Or 'BLENDER_EEVEE' for a different render engine.
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = imgPATH

# Change the background 
# bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)

# Set up the camera
# This example places the camera and points it toward the center of the scene:
camera = bpy.data.cameras.new("Camera")
camera_obj = bpy.data.objects.new("Camera", camera)
scene.collection.objects.link(camera_obj)
scene.camera = camera_obj
camera_obj.location = (0, -3, 2)
camera_obj.rotation_euler = (1.0, 0, 1.0)

# Make sure the imported object is centered and fits nicely in the camera view
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.view3d.camera_to_view_selected()

# Set up lighting
light_data = bpy.data.lights.new(name="light_2.80", type='POINT')
light_object = bpy.data.objects.new(name="light_2.80", object_data=light_data)
scene.collection.objects.link(light_object)
light_object.location = (0, 0, 2)

# Render the scene
bpy.ops.render.render(write_still=True)