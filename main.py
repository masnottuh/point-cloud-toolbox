####################################################################
# Robert Hutton - UNR Dept of Mech E - rhutton@unr.edu
####################################################################
from utils import *
from pointCloudToolbox import *
import os
##################################
logging.basicConfig(level=logging.INFO)
##################################

#Validate geometries
shapes =  generate_pv_shapes()

# List of shape names corresponding to the shapes generated
shape_names = ["plane", "sphere", "cylinder", "torus", "egg_carton"]

# Iterate over the shapes and save points as .ply with descriptive filenames
for shape, name in zip(shapes, shape_names):
    points = shape.points  # Extract points from the shape
    filename = f"test_shapes/{name}.ply"  # Create a filename based on the shape's name
    save_points_to_ply(points, filename)  # Save points to .ply file
    
    # Optionally load and plot the saved shape
    pts = pv.read(filename)
    pts.plot()

file_directory = "./test_shapes/"
files = [os.path.join(file_directory, file) for file in os.listdir(file_directory) if file.endswith(".ply")]

for file in files:
    validate_shape(str(file))
    