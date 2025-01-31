####################################################################
# Robert Hutton - UNR Dept of Mech E - rhutton@unr.edu
####################################################################
####################################################################
# FOR ANALYZING SCANS OF SHEETS
####################################################################
from utils import *
from pointCloudToolbox import *
import os
import logging
import pandas as pd
import subprocess
import sys
from scipy.integrate import dblquad
import glob

##################################
logging.basicConfig(level=logging.INFO)
##################################

output_dir = './output'  # Output directory
test_shapes_dir = './Scans'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(test_shapes_dir):
    os.makedirs(test_shapes_dir)

# Storage for results
results = []

# .ply files in the directory
existing_ply_files = glob.glob(f"{test_shapes_dir}/*.ply")


 # Process .ply files
for filepath in existing_ply_files:
    shape_name = os.path.basename(filepath).split(".")[0]
    logging.info(f"Processing existing .ply file: {shape_name}")

    # Process the existing file
    try:
        bending_energy, stretching_energy, computed_area = validate_shape(filepath)
        logging.info(f"Processed {shape_name}: Bending Energy: {bending_energy}, Stretching Energy: {stretching_energy}, Computed Area: {computed_area}")
    except Exception as e:
        logging.error(f"Error processing {shape_name}: {e}")
        bending_energy, stretching_energy, computed_area = "Error", "Error", "Error"

    # Append results (no theoretical area for pre-existing shapes)
    results.append({
        "Shape": shape_name,
        "Num Points": "N/A",
        "Computed Area": computed_area,
        "Bending Energy": bending_energy,
        "Stretching Energy": stretching_energy
    })

# # Save results to a DataFrame and display
results_df = pd.DataFrame(results)

# Save results to CSV
results_df.to_csv(f"shape_comparison_results.csv", index=False)