####################################################################
# Robert Hutton - UNR Dept of Mech E - rhutton@unr.edu
####################################################################
from utils import *
from pointCloudToolbox import *
import os
##################################
logging.basicConfig(level=logging.INFO)
##################################
import sys
sys.setrecursionlimit(10**9) 

# Create test shapes
shapes =  generate_pv_shapes(num_points=10000, perturbation_strength=0.05)

# List of shape names corresponding to the shapes generated
shape_names = ["plane", "plane_perturbed", "sphere", "sphere_perturbed", "cylinder",
 "cylinder_perturbed", "torus", "torus_perturbed", "egg_carton", "egg_carton_perturbed"]

# Iterate over the shapes and save points as .ply with descriptive filenames
for shape, shape_names in zip(shapes, shape_names):
    points = shape.points  # Extract points from the shape
    filename = f"test_shapes/{shape_names}.ply"  # Create a filename based on the shape's name
    save_points_to_ply(points, filename)  # Save points to .ply file
    
    # Optionally load and plot the saved shape
    pts = pv.read(filename)
    pts.plot(text=f'{shape_names}', off_screen=True)

file_directory = "./test_shapes/"
files = [os.path.join(file_directory, file) for file in os.listdir(file_directory) if file.endswith(".ply")]

for file in files:
    validate_shape(str(file))
    
