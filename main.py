####################################################################
# Robert Hutton - UNR Dept of Mech E - rhutton@unr.edu
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
test_shapes_dir = './test_shapes'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(test_shapes_dir):
    os.makedirs(test_shapes_dir)

# Parameters for testing
radii = [0.1, 1.0, 10.0, 100.0, 1000]  # Radii for shapes
point_densities = [5000, 10000, 25000, 50000]  # Number of points for each shape

# Storage for results
results = []

# Function definitions for surface area integration
def egg_carton_surface_element(x, y):
    """Calculate the surface element for the egg carton function."""
    dzdx = np.cos(x) * np.cos(y)  # Partial derivative with respect to x
    dzdy = -np.sin(x) * np.sin(y)  # Partial derivative with respect to y
    return np.sqrt(1 + dzdx**2 + dzdy**2)


# Check for existing .ply files in the directory
existing_ply_files = glob.glob(f"{test_shapes_dir}/*.ply")

# Loop through radii and point densities
for radius in radii:
    for num_points in point_densities:
        logging.info(f"Testing radius: {radius}, num_points: {num_points}")

        # Generate new shapes
        perturbation_strength = 0.0001*np.sqrt(radius)  # perturbation

        shapes = generate_pv_shapes(num_points=num_points, perturbation_strength=perturbation_strength, radius=radius)
        shape_names = ["cylinder", "cylinder_perturbed", "torus", "torus_perturbed", "sphere", "sphere_perturbed", "egg_carton", "egg_carton_perturbed"]

        # Process generated shapes
        for shape, shape_name in zip(shapes, shape_names):
            points = shape.points
            filename = f"{test_shapes_dir}/{shape_name}_radius_{radius}_points_{num_points}.ply"
            save_points_to_ply(points, filename)

            # Process the saved shape
            loaded_shape = pv.read(filename)
            try:
                bending_energy, stretching_energy, computed_area = validate_shape(filename)
                logging.info(f"Processed {shape_name}: Bending Energy: {bending_energy}, Stretching Energy: {stretching_energy}, Computed Area: {computed_area}")
            except Exception as e:
                logging.error(f"Error processing {shape_name}: {e}")
                bending_energy, stretching_energy, computed_area = "Error", "Error", "Error"

            # Calculate theoretical area
            theoretical_area = None
            if "sphere" in shape_name:
                theoretical_area = 4.0 * 3.14159 * (radius ** 2.0)
            elif "cylinder" in shape_name:
                height = 2 * radius
                theoretical_area = 2.0 * ((3.14159 * radius) * height)
            elif "torus" in shape_name:
                tube_radius = radius
                cross_section_radius = radius / 3
                theoretical_area = (2 * 3.14159 * tube_radius) * (2 * 3.14159 * cross_section_radius)
            elif "egg_carton" in shape_name:
                theoretical_area, _ = dblquad(egg_carton_surface_element, -3, 3, lambda x: -3, lambda x: 3)

            # Append results
            results.append({
                "Shape": shape_name,
                "Radius": radius,
                "Num Points": num_points,
                "Theoretical Area": theoretical_area,
                "Computed Area": computed_area,
                "Bending Energy": bending_energy,
                "Stretching Energy": stretching_energy
            })

        # Process existing .ply files
# for filepath in existing_ply_files:
#     shape_name = os.path.basename(filepath).split(".")[0]
#     logging.info(f"Processing existing .ply file: {shape_name}")

#     # Process the existing file
#     try:
#         bending_energy, stretching_energy, computed_area = validate_shape(filepath)
#         logging.info(f"Processed {shape_name}: Bending Energy: {bending_energy}, Stretching Energy: {stretching_energy}, Computed Area: {computed_area}")
#     except Exception as e:
#         logging.error(f"Error processing {shape_name}: {e}")
#         bending_energy, stretching_energy, computed_area = "Error", "Error", "Error"

    # Append results (no theoretical area for pre-existing shapes)
    results.append({
        "Shape": shape_name,
        "Radius": "N/A",
        "Num Points": "N/A",
        "Theoretical Area": "N/A",
        "Computed Area": computed_area,
        "Bending Energy": bending_energy,
        "Stretching Energy": stretching_energy
    })

# print("Completed testing for radius:", radius, "and num_points:", num_points)

# Save results to a DataFrame and display
results_df = pd.DataFrame(results)

# Ensure 'Theoretical Area' is numeric, replacing non-numeric values with NaN
results_df['Theoretical Area'] = pd.to_numeric(results_df['Theoretical Area'], errors='coerce')

# Save results to CSV
results_df.to_csv(f"shape_comparison_results.csv", index=False)

# Display results in the console for verification
print("Shape Comparison Results:")
results_df

# Optional: Visualize results (can be commented out for non-visual environments)
for shape in results_df['Shape'].unique():
    df_shape = results_df[results_df['Shape'] == shape].copy()
    
    # Drop rows where 'Theoretical Area' is NaN
    df_shape = df_shape.dropna(subset=['Theoretical Area'])
    
    if not df_shape.empty:  # Ensure there's data to plot
        plt.figure()
        plt.title(f"{shape} Area Comparison")
        plt.plot(df_shape['Num Points'], df_shape['Theoretical Area'], label='Theoretical', marker='o')
        plt.plot(df_shape['Num Points'], df_shape['Computed Area'], label='Computed', marker='x')
        plt.xlabel('Number of Points')
        plt.ylabel('Surface Area')
        plt.legend()
        plt.savefig(f"{output_dir}/{shape}_comparison.png")
        plt.show()
    else:
        print(f"Skipping plot for {shape} due to missing data.")