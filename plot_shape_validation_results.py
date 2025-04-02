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
                          (df_filtered["Num Points"] >= 1000) & 
                          (df_filtered["Perturbed"] == False) & 
                          (df_filtered["Num Points"] <= 1_500_000)]

# Add Base Shape column
df_filtered["Base Shape"] = df_filtered["Shape"].str.replace(r"_Perturbed|_Unperturbed", "", regex=True)

# Compute theoretical curvatures correctly
def compute_theoretical_curvatures(shape, radius):
    shape = shape.lower()
    if shape == "sphere":
        mean = 1/radius
        gaussian = 1/(radius**2)
    elif shape == "cylinder":
        mean = 1/(2*radius)
        gaussian = 0
    elif shape == "torus":
        R, r = radius, radius / 3  # Example major and minor radii, adjust as needed
        mean = (R + 2*r) / (2*r*(R + r))
        gaussian = np.cos(0)/(r*(R + r))  # Assuming outer curvature point, adjust as needed
    elif shape == "egg_carton":
        mean = 0  # Mean curvature varies, set as needed
        gaussian = -1/(radius**2)
    else:
        mean = gaussian = None
    return mean, gaussian

# Apply theoretical curvature calculations
df_filtered[["Theoretical Mean Curvature", "Theoretical Gaussian Curvature"]] = df_filtered.apply(
    lambda row: compute_theoretical_curvatures(row["Base Shape"], row["Radius"]), axis=1, result_type="expand"
)

# Directories to save plots
scatter_output_dir = "scatter_plots"
histogram_output_dir = "curvature_histograms"
os.makedirs(scatter_output_dir, exist_ok=True)
os.makedirs(histogram_output_dir, exist_ok=True)

# Generate scatter plots for Percent Error vs everything else separately for each Base Shape
column_pairs = [(col, "Percent Error") for col in numeric_columns if col != "Percent Error"]
log = True

for base_shape, df_shape in df_filtered.groupby('Base Shape'):
    for x_col, y_col in column_pairs:
        df_shape_subset = df_shape.dropna(subset=[x_col, y_col])

        if df_shape_subset.empty:
            continue

        plt.figure(figsize=(12, 6))

        shape_radius_combinations = df_shape_subset[['Radius']].drop_duplicates().sort_values('Radius').reset_index(drop=True)
        num_colors = len(shape_radius_combinations)
        color_map = cm.get_cmap('tab10', num_colors)

        color_dict = {radius: color_map(i) for i, radius in enumerate(shape_radius_combinations['Radius'])}

        for radius, group in df_shape_subset.groupby("Radius"):
            group = group.sort_values(by="Num Points")
            plt.plot(group[x_col], group[y_col], marker="s", linestyle="-", alpha=0.7,
                     color=color_dict[radius], label=f"R={radius}")

        if log:
            plt.xscale("log")
            plt.yscale("log")

        plt.xticks(rotation=45, ha="right")
        plt.xlabel(x_col, fontsize=18)
        plt.ylabel(y_col, fontsize=18)
        plt.title(f"{y_col} vs {x_col} for {base_shape.capitalize()} (Log-Log)", fontsize=20)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(title="Radius", loc="best", fontsize=16, frameon=True)
        plt.tight_layout()

        safe_x_col = x_col.replace('/', '_').replace(' ', '_')
        safe_y_col = y_col.replace('/', '_').replace(' ', '_')
        scatter_filename = os.path.join(scatter_output_dir, f'{base_shape}_{safe_y_col}_vs_{safe_x_col}.png')
        plt.savefig(scatter_filename)
        plt.clf()
        plt.close()

# Generate histograms for mean and Gaussian curvatures
for idx, row in df_filtered.iterrows():
    base_shape = row['Base Shape']
    radius = row['Radius']
    num_points = int(row['Num Points'])
    variant = 'Perturbed' if row.get('Perturbed', False) else 'Unperturbed'

    # Handle common variations in filename point count
    possible_points = [num_points]
    if base_shape == 'egg_carton':
        if num_points > 100000:  # known precision offset
            possible_points.append(num_points - (num_points % 100000))
        possible_points.append(961)
    elif base_shape == 'torus':
        if num_points > 100000:
            possible_points.append(100489)

    found_file = False
    for pts in possible_points:
        mean_curvature_file = f"curvature_data/{base_shape}_{variant}_radius_{radius}_points_{pts}_mean.npy"
        gaussian_curvature_file = f"curvature_data/{base_shape}_{variant}_radius_{radius}_points_{pts}_gaussian.npy"

        if os.path.exists(mean_curvature_file) and os.path.exists(gaussian_curvature_file):
            found_file = True
            break

    if not found_file:
        print(f"Curvature data not found for {base_shape}, radius {radius}, points {num_points}")
        continue

    mean_curvatures = np.load(mean_curvature_file)
    gaussian_curvatures = np.load(gaussian_curvature_file)

    for curvature, name, theoretical_value in [(mean_curvatures, 'Mean Curvature', row['Theoretical Mean Curvature']),
                                               (gaussian_curvatures, 'Gaussian Curvature', row['Theoretical Gaussian Curvature'])]:
        plt.figure(figsize=(10, 6))
        plt.hist(curvature, bins=100, density=True, alpha=0.7, color='blue')
        if theoretical_value is not None:
            plt.axvline(theoretical_value, color='red', linestyle='--', linewidth=2, label=f'Theoretical: {theoretical_value:.4f}')

        plt.xlabel(name, fontsize=18)
        plt.ylabel('Density', fontsize=18)
        plt.title(f'{name} Histogram for {base_shape.capitalize()} (R={radius}, Points={pts})', fontsize=20)
        plt.legend(fontsize=16)
        plt.grid(True, linestyle='--')
        plt.tight_layout()

        histogram_filename = os.path.join(histogram_output_dir, f'{base_shape}_{name.replace(" ", "_")}_radius_{radius}_points_{pts}_histogram.png')
        plt.savefig(histogram_filename)
        plt.clf()
        plt.close()

print("Scatter plots and histograms saved successfully.")