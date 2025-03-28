import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

# Load and preprocess data
file_path = "incremental_shape_comparison_results.csv"
df = pd.read_csv(file_path)

# Remove rows with "Error"
df_filtered = df[~df.apply(lambda row: row.astype(str).str.contains("Error", na=False).any(), axis=1)].copy()

# Convert necessary columns
numeric_columns = ["Computed Area", "Percent Error", "Point Density", "Radius", "Num Points", "Theoretical Area"]
df_filtered.loc[:, numeric_columns] = df_filtered[numeric_columns].apply(pd.to_numeric, errors="coerce")

# Further filtering
df_filtered = df_filtered[(df_filtered["Percent Error"] <= 100) & 
                          (df_filtered["Perturbed"] == False) & 
                          (df_filtered["Num Points"] >= 1000) & 
                          (df_filtered["Num Points"] <= 1_500_000)]

# Add Base Shape column
df_filtered["Base Shape"] = df_filtered["Shape"].str.replace(r"_Perturbed|_Unperturbed", "", regex=True)

# Compute theoretical curvatures
def compute_theoretical_curvatures(shape, radius):
    shape = shape.lower()
    if shape == "sphere":
        return 1/radius, (1 / radius) ** 2
    elif shape == "cylinder":
        return 1/(2*radius), (1 / (2 * radius)) ** 2
    elif shape == "torus":
        return 2/(3*radius), ((2 / (3 * radius)) ** 2)
    elif shape == "egg_carton":
        return 5/radius, ((5 / radius) ** 2)
    else:
        return None, None

# Apply theoretical curvature calculations
df_filtered[["Theoretical Mean Curvature", "Theoretical Gaussian Curvature"]] = df_filtered.apply(
    lambda row: compute_theoretical_curvatures(row["Base Shape"], row["Radius"]), axis=1, result_type="expand"
)

# Directories to save plots
scatter_output_dir = "scatter_plots"
histogram_output_dir = "curvature_histograms"
os.makedirs(scatter_output_dir, exist_ok=True)
os.makedirs(histogram_output_dir, exist_ok=True)

# Generate histograms for mean and Gaussian curvatures
for idx, row in df_filtered.iterrows():
    base_shape = row['Base Shape']
    radius = row['Radius']
    num_points = int(row['Num Points'])
    variant = 'Unperturbed'

    mean_curvature_file = f"curvature_data/{base_shape}_{variant}_radius_{radius}_points_{num_points}_mean.npy"
    gaussian_curvature_file = f"curvature_data/{base_shape}_{variant}_radius_{radius}_points_{num_points}_gaussian.npy"

    if not os.path.exists(mean_curvature_file) or not os.path.exists(gaussian_curvature_file):
        print(f"Curvature data not found for {base_shape}, radius {radius}, points {num_points}")
        continue

    mean_curvatures = np.load(mean_curvature_file)
    gaussian_curvatures = np.load(gaussian_curvature_file)

    for curvature, name, theoretical_value in [(mean_curvatures, 'Mean Curvature', row['Theoretical Mean Curvature']),
                                               (gaussian_curvatures, 'Gaussian Curvature', row['Theoretical Gaussian Curvature'])]:
        plt.figure(figsize=(10, 6))
        plt.hist(curvature, bins=100, density=True, alpha=0.7, color='blue')
        plt.axvline(theoretical_value, color='red', linestyle='--', linewidth=2, label=f'Theoretical: {theoretical_value:.4f}')

        plt.xlabel(name, fontsize=18)
        plt.ylabel('Density', fontsize=18)
        plt.title(f'{name} Histogram for {base_shape.capitalize()} (R={radius}, Points={num_points})', fontsize=20)
        plt.legend(fontsize=16)
        plt.grid(True, linestyle='--')
        plt.tight_layout()

        histogram_filename = os.path.join(histogram_output_dir, f'{base_shape}_{name.replace(" ", "_")}_radius_{radius}_points_{num_points}_histogram.png')
        plt.savefig(histogram_filename)
        plt.clf()
        plt.close()

print("Histograms saved successfully.")
