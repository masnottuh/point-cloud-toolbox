import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Load the CSV file
file_path = "incremental_shape_comparison_results.csv"  # Update this path as needed
df = pd.read_csv(file_path)

# Remove rows where any column contains the string "Error"
df_filtered = df[~df.apply(lambda row: row.astype(str).str.contains("Error", na=False).any(), axis=1)].copy()

# Convert necessary columns to numeric values, handling errors
numeric_columns = ["Computed Area", "Percent Error", "Point Density", "Radius", "Num Points", "Theoretical Area"]
df_filtered.loc[:, numeric_columns] = df_filtered[numeric_columns].apply(pd.to_numeric, errors="coerce")

# Remove rows where Percent Error is greater than 10%
df_filtered = df_filtered[df_filtered["Percent Error"] <= 100]

# Remove perturbed rows
df_filtered = df_filtered[df_filtered["Perturbed"] == False]

# Ensure 'Num Points' is within the expected range
df_filtered = df_filtered[(df_filtered["Num Points"] >= 1000) & (df_filtered["Num Points"] <= 1_500_000)]

# Extract 'Base Shape' from the 'Shape' column
if "Shape" in df_filtered.columns:
    df_filtered["Base Shape"] = df_filtered["Shape"].str.replace(r"_Perturbed|_Unperturbed", "", regex=True)
else:
    print("Error: 'Shape' column is missing! Cannot create 'Base Shape'.")
    exit()

# Compute Theoretical Mean Curvature Squared
def compute_theoretical_mean_curvature_squared(shape, radius):
    if pd.isna(radius) or pd.isna(shape):
        return None  # Avoid errors if shape or radius is missing
    
    shape = str(shape).lower().strip()  # Normalize shape names
    radius = float(radius)  # Ensure radius is a float

    if shape == "sphere":
        return (1 / radius) ** 2  # Mean curvature squared: H² = (1/r)²
    elif shape == "cylinder":
        return (1 / (2 * radius)) ** 2  # Mean curvature squared: H² = (1/2r)²
    elif shape == "torus":
        return ((2 / (3 * radius)) ** 2)  # Approximate mean curvature squared for torus
    elif shape == "egg_carton":
        return ((5 / radius) ** 2)  # Approximate mean curvature squared for egg_carton
    else:
        print(f"Warning: Unexpected shape '{shape}' encountered. Assigning None.")
        return None  # Handle unexpected shape names

# Apply the function to compute theoretical curvature
df_filtered["Theoretical Mean Curvature²"] = df_filtered.apply(
    lambda row: compute_theoretical_mean_curvature_squared(row["Base Shape"], row["Radius"]), axis=1
)

# Debugging: Check computed values
print("Unique values in 'Theoretical Mean Curvature²':", df_filtered["Theoretical Mean Curvature²"].dropna().unique())

# Create derived features
df_filtered["Num Points / Radius"] = df_filtered["Num Points"] / df_filtered["Radius"]
df_filtered["Num Points × Theoretical Curvature²"] = df_filtered["Num Points"] * df_filtered["Theoretical Mean Curvature²"]

# Ensure valid values (remove NaNs and infinities)
df_filtered = df_filtered.replace([float('inf'), -float('inf')], None).dropna(
    subset=["Num Points / Radius", "Num Points × Theoretical Curvature²", "Percent Error"]
)

# Add derived columns to numeric list
numeric_columns += ["Theoretical Mean Curvature²", "Num Points / Radius", "Num Points × Theoretical Curvature²"]

# Ensure Percent Error is always on the y-axis
column_pairs = [(x_col, "Percent Error") for x_col in numeric_columns if x_col != "Percent Error"]

log = True  # Enable log-log scaling

# Ensure modifications don't affect original data
df_filtered = df_filtered.copy()
df_filtered.loc[:, numeric_columns] = df_filtered[numeric_columns].apply(pd.to_numeric, errors="coerce")

# Generate scatter plots for Percent Error vs everything else
for x_col, y_col in column_pairs:
    if df_filtered.empty:
        print(f"Warning: df_filtered is empty. Skipping plot for {y_col} vs {x_col}.")
        continue
    
    if x_col not in df_filtered.columns or y_col not in df_filtered.columns:
        print(f"Warning: Column {x_col} or {y_col} not found in df_filtered. Skipping plot.")
        continue

    df_filtered_subset = df_filtered.dropna(subset=[x_col, y_col])

    num_unique_x = df_filtered_subset[x_col].nunique()
    num_unique_y = df_filtered_subset[y_col].nunique()

    if num_unique_x < 2 or num_unique_y < 2:
        print(f"Warning: Not enough unique values in {x_col} or {y_col} for a meaningful plot. Skipping.")
        continue  

    plt.figure(figsize=(12, 6))

    # Find all unique shape-radius combinations
    shape_radius_combinations = df_filtered_subset[['Base Shape', 'Radius']].drop_duplicates()
    shape_radius_combinations = shape_radius_combinations.sort_values(['Base Shape', 'Radius']).reset_index(drop=True)

    num_colors = len(shape_radius_combinations)
    color_map = cm.get_cmap('tab20', num_colors)  # You can choose other colormaps like 'tab20', 'hsv', 'jet', etc.

    # Create a mapping from (shape, radius) to color
    color_dict = {
        (row['Base Shape'], row['Radius']): color_map(i)
        for i, row in shape_radius_combinations.iterrows()
    }

    # Plot each group separately
    for perturbed_status, df_perturbed in df_filtered_subset.groupby("Perturbed"):
        marker_style = "o" if perturbed_status else "s"
        linestyle = "--" if perturbed_status else "-"

        for (shape, radius), group in df_perturbed.groupby(["Base Shape", "Radius"]):
            group = group.sort_values(by="Num Points")

            plt.plot(group[x_col], group[y_col],
                     marker=marker_style,
                     linestyle=linestyle,
                     alpha=0.7,
                     color=color_dict[(shape, radius)],
                     label=f"{shape} (R={radius}, {'Perturbed' if perturbed_status else 'Unperturbed'})")

    if log:
        plt.xscale("log")
        plt.yscale("log")

        x_min, x_max = df_filtered_subset[x_col].min(), df_filtered_subset[x_col].max()
        y_min, y_max = df_filtered_subset[y_col].min(), df_filtered_subset[y_col].max()

        x_offset = (x_max / x_min) ** 0.05 if x_min > 0 else 1.1
        y_offset = (y_max / y_min) ** 0.05 if y_min > 0 else 1.1

        plt.xlim(x_min / x_offset, x_max * x_offset)
        plt.ylim(y_min / y_offset, y_max * y_offset)

    plt.xticks(rotation=45, ha="right")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col} (Log-Log Scale) - Perturbed & Unperturbed Trends")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(title="Shape, Radius & Perturbation", loc="best", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.show()

    plt.clf()
    plt.close()
