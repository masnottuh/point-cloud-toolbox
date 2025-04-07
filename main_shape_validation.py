from utils import *
from pointCloudToolbox import *
import os
import logging
import pandas as pd
import subprocess
import sys
from scipy.integrate import dblquad
import glob
from wakepy import keep
import numpy as np

def compute_density_for_target_points(radius, target_points, shape_area_func):
    """Calculate the density required to achieve the target number of points."""
    area = shape_area_func(radius)
    return target_points / area if area > 0 else None

def compute_egg_carton_surface_area(radius):
    """Compute the surface area of the egg-carton function using numerical integration."""
    def egg_carton_surface_element(x, y):
        z_scale = radius / 10.0
        dzdx = z_scale * (np.pi / radius) * np.cos(x / radius * np.pi) * np.cos(y / radius * np.pi)
        dzdy = -z_scale * (np.pi / radius) * np.sin(x / radius * np.pi) * np.sin(y / radius * np.pi)
        return np.sqrt(1 + dzdx**2 + dzdy**2)

    area, _ = dblquad(egg_carton_surface_element, -radius, radius, lambda x: -radius, lambda x: radius)
    return area

theoretical_bending_energy_funcs = {
    "sphere": lambda r: 4 * np.pi,
    "cylinder": lambda r: np.pi,
    "torus": lambda r: np.nan,        # requires numeric integration
    "egg_carton": lambda r: np.nan    # requires numeric integration
}

theoretical_stretching_energy_funcs = {
    "sphere": lambda r: 4 * np.pi,
    "cylinder": lambda r: 0,
    "torus": lambda r: 0,
    "egg_carton": lambda r: np.nan    # requires numeric integration
}


# Keep awake for long tests
with keep.running():
    csv_filename = "incremental_shape_comparison_results.csv"
    csv_exists = os.path.exists(csv_filename)

    logging.basicConfig(level=logging.INFO)

    output_dir = './output'
    test_shapes_dir = './test_shapes'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(test_shapes_dir, exist_ok=True)

    radius_values = [1, 10, 100, 1000]
    target_num_points = [1000000]

    results = []

    for radius in radius_values:
        shape_area_funcs = {
            "sphere": lambda r: 4.0 * np.pi * (r ** 2.0),
            "cylinder": lambda r: 2.0 * (np.pi * r * (2 * r)),
            "torus": lambda r: (2 * np.pi * r) * (2 * np.pi * (r / 3)),
            "egg_carton": compute_egg_carton_surface_area
        }

        for shape_name, shape_area_func in shape_area_funcs.items():
            densities = [compute_density_for_target_points(radius, n, shape_area_func) for n in target_num_points]
            densities = [d for d in densities if d is not None]

            for density, num_points in zip(densities, target_num_points):
                logging.info(f"Testing {shape_name} with radius {radius}, density {density}, num_points {num_points}")

                theoretical_area = shape_area_func(radius)
                theoretical_bending_energy = theoretical_bending_energy_funcs[shape_name](radius)
                theoretical_stretching_energy = theoretical_stretching_energy_funcs[shape_name](radius)

                perturbation_strength = 0.001 * np.sqrt(theoretical_area)

                shape, shape_perturbed = generate_pv_shapes(
                    shape_name, num_points=num_points, perturbation_strength=perturbation_strength, radius=radius
                )

                shape_filename = f"{test_shapes_dir}/{shape_name}_radius_{radius}_points_{num_points}.ply"
                shape_perturbed_filename = f"{test_shapes_dir}/{shape_name}_radius_{radius}_points_{num_points}_perturbed.ply"

                save_points_to_ply(np.asarray(shape.points), shape_filename)
                save_points_to_ply(np.asarray(shape_perturbed.points), shape_perturbed_filename)

                for variant, filename, perturbed_flag in [("Unperturbed", shape_filename, False), 
                                                        ("Perturbed", shape_perturbed_filename, True)]:
                    try:
                        bending_energy, stretching_energy, computed_area = validate_shape(filename, "N", shape_name, variant, radius)
                    except Exception as e:
                        logging.error(f"Error processing {shape_name} ({variant}): {e}")
                        bending_energy, stretching_energy, computed_area = "Error", "Error", "Error"

                    computed_area = float(computed_area) if computed_area != "Error" else float('nan')
                    percent_error = 100 * abs((theoretical_area - computed_area) / theoretical_area) if theoretical_area > 0 else float('nan')

                    results.append({
                        "Shape": f"{shape_name}_{variant}",
                        "Radius": radius,
                        "Num Points": num_points,
                        "Point Density": density,
                        "Theoretical Area": theoretical_area,
                        "Computed Area": computed_area,
                        "Percent Error": percent_error,
                        "Bending Energy": bending_energy,
                        "Stretching Energy": stretching_energy,
                        "Theoretical Bending Energy": theoretical_bending_energy,
                        "Theoretical Stretching Energy": theoretical_stretching_energy,
                        "Perturbed": perturbed_flag
                    })

                    results_df = pd.DataFrame([results[-1]])
                    results_df.to_csv(csv_filename, mode='a', header=not csv_exists, index=False)
                    csv_exists = True

    print("Testing completed.")
    pd.DataFrame(results).to_csv("backup_shape_comparison_results.csv", index=False)
