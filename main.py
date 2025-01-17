####################################################################
# Robert Hutton - UNR Dept of Mech E - rhutton@unr.edu
####################################################################
from utils import *
from pointCloudToolbox import *
import os
import logging
import pandas as pd
import pyvista as pv

##################################
logging.basicConfig(level=logging.INFO)
##################################

output_dir = './output'  # Output directory
test_shapes_dir = './test_shapes'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(test_shapes_dir):
    os.makedirs(test_shapes_dir)

# Parameters for testing
radii = [5.0, 10.0]  # Radii for shapes
point_densities = [5000]  # Number of points for each shape
perturbation_strength = 0.0  # No perturbation for correctness check

# Storage for results
results = []

# Loop through radii and point densities
for radius in radii:
    for num_points in point_densities:
        logging.info(f"Testing radius: {radius}, num_points: {num_points}")
        
        # Create test shapes
        shapes = generate_pv_shapes(num_points=num_points, perturbation_strength=perturbation_strength)
        # shape_names = ["sphere", "cylinder", "torus"]
        shape_names = ["cylinder"]

        # Iterate over the shapes and save points as .ply with descriptive filenames
        for shape, shape_name in zip(shapes, shape_names):
            points = shape.points  # Extract points from the shape
            filename = f"{test_shapes_dir}/{shape_name}_radius_{radius}_points_{num_points}.ply"  # Filename based on parameters
            save_points_to_ply(points, filename)  # Save points to .ply file
            
            # Optionally load and validate the saved shape
            loaded_shape = pv.read(filename)
            validate_shape(str(filename))  # Retain validate_shape call

            # Calculate and compare theoretical vs computed surface areas
            if "sphere" in shape_name:
                theoretical_area = 4.0 * 3.14159 * (radius ** 2.0)
            elif "cylinder" in shape_name:
                height = 2 * radius  # Assume height equals diameter for simplicity
                theoretical_area = (2.0 * 3.14159 * radius * height)
            elif "torus" in shape_name:
                tube_radius = radius
                cross_section_radius = radius / 3  # Assume proportional tube radius
                theoretical_area = (2 * 3.14159 * tube_radius) * (2 * 3.14159 * cross_section_radius)
            # elif "egg_carton" in shape_name:
            #     theoretical_area = "N/A (non-standard shape)"
            
            computed_area = loaded_shape.area
            results.append({
                "Shape": shape_name,
                "Radius": radius,
                "Num Points": num_points,
                "Theoretical Area": theoretical_area,
                "Computed Area": computed_area
            })

        print("Completed testing for radius:", radius, "and num_points:", num_points)

# Save results to a DataFrame and display
results_df = pd.DataFrame(results)
results_df.to_csv(f"{output_dir}/shape_comparison_results.csv", index=False)

# Display results in the console for verification
print("Shape Comparison Results:")
print(results_df)

# Optional: Visualize results (can be commented out for non-visual environments)
# import matplotlib.pyplot as plt
# for shape in results_df['Shape'].unique():
#     df_shape = results_df[results_df['Shape'] == shape]
#     plt.figure()
#     plt.title(f"{shape} Area Comparison")
#     plt.plot(df_shape['Num Points'], df_shape['Theoretical Area'], label='Theoretical', marker='o')
#     plt.plot(df_shape['Num Points'], df_shape['Computed Area'], label='Computed', marker='x')
#     plt.xlabel('Number of Points')
#     plt.ylabel('Surface Area')
#     plt.legend()
#     plt.savefig(f"{output_dir}/{shape}_comparison.png")
