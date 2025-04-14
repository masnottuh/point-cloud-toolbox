from utils import *
from pointCloudToolbox import *
import os
import time
import logging
import pandas as pd
import subprocess
import sys
from scipy.integrate import dblquad
import glob
from wakepy import keep
import numpy as np
import gc


def compute_density_for_target_points(radius, target_points, shape_area_func):
    """Calculate the density required to achieve the target number of points."""
    area = shape_area_func(radius)
    return target_points / area if area > 0 else None

def compute_egg_carton_surface_area(radius):
    """Compute the surface area of the egg-carton function using numerical integration."""
    def egg_carton_surface_element(x, y):
        # Here we assume the egg carton function z = 0.1 * sin(pi*x) * cos(pi*y)
        # and we scale the z-coordinate with radius/10 so that the domain becomes [-radius, radius].
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

    # List of target numbers of points to test (outer loop).
    target_num_points = [200000, 300000, 400000, 500000, 1000000]  # You can add more values here if needed.
    # List of radii values to test.
    radius_values = [0.1, 10, 1000]

    # Define the shape area functions that depend on a radius.
    shape_area_funcs = {
        "sphere": lambda r: 4.0 * np.pi * (r ** 2.0),
        "cylinder": lambda r: 2.0 * (np.pi * r * (2 * r)),  # Lateral surface: 2πr (height 2r)
        "torus": lambda r: (2 * np.pi * r) * (2 * np.pi * (r / 3)),  # (2π·major_radius) * (2π·tube_radius)
        "egg_carton": lambda r: compute_egg_carton_surface_area(r)
    }

    results = []

    # Outer loop: iterate over each target number of points.
    for num_points in target_num_points:

        gc.collect()
        print("overhead collected and cleared (memory optimized)")

        logging.info(f"Processing target number of points: {num_points}")

        # Loop over each radius.
        for radius in radius_values:
            # For each target, iterate over all shape types.
            for shape_name, shape_area_func in shape_area_funcs.items():
                density = compute_density_for_target_points(radius, num_points, shape_area_func)
                if density is None:
                    continue

                logging.info(f"Testing {shape_name} with radius {radius}, density {density}, num_points {num_points}")

                theoretical_area = shape_area_func(radius)
                theoretical_bending_energy = theoretical_bending_energy_funcs[shape_name](radius)
                theoretical_stretching_energy = theoretical_stretching_energy_funcs[shape_name](radius)

                perturbation_strength = 0.001 * np.sqrt(theoretical_area)

                # Generate the shape (unperturbed version only).
                shape, _ = generate_pv_shapes(
                    shape_name, num_points=num_points, perturbation_strength=perturbation_strength, radius=radius
                )

                shape_filename = f"{test_shapes_dir}/{shape_name}_radius_{radius}_points_{num_points}.ply"
                save_points_to_ply(np.asarray(shape.points), shape_filename)

                # Process only the "Unperturbed" variant.
                for variant, filename, perturbed_flag in [("Unperturbed", shape_filename, False)]:
                    # Record the start time for validation.
                    t_start = time.time()
                    try:
                        bending_energy, stretching_energy, computed_area = validate_shape(
                            filename, "N", shape_name, variant, radius
                        )
                    except Exception as e:
                        logging.error(f"Error processing {shape_name} ({variant}): {e}")
                        bending_energy, stretching_energy, computed_area = "Error", "Error", "Error"

                    run_time = time.time() - t_start

                    computed_area = float(computed_area) if computed_area != "Error" else float('nan')
                    percent_area_error = (
                        100 * abs((theoretical_area - computed_area) / theoretical_area)
                        if theoretical_area > 0 else float('nan')
                    )

                    # Handle case where energies are not valid numbers.
                    if bending_energy == "Error" or stretching_energy == "Error":
                        percent_error_bending = float("nan")
                        percent_error_stretching = float("nan")
                    else:
                        # Compute percent error for bending energy.
                        if theoretical_bending_energy != 0:
                            percent_error_bending = (
                                100 * abs(theoretical_bending_energy - bending_energy) / abs(theoretical_bending_energy)
                            )
                        else:
                            percent_error_bending = abs(theoretical_bending_energy - bending_energy)
                        # Compute percent error for stretching energy.
                        if theoretical_stretching_energy != 0:
                            percent_error_stretching = (
                                100 * abs(theoretical_stretching_energy - stretching_energy) / abs(theoretical_stretching_energy)
                            )
                        else:
                            percent_error_stretching = abs(theoretical_stretching_energy - stretching_energy)

                    results.append({
                        "Shape": f"{shape_name}_{variant}",
                        "Radius": radius,
                        "Num Points": num_points,
                        "Point Density": density,
                        "Theoretical Area": theoretical_area,
                        "Computed Area": computed_area,
                        "Percent Area Error": percent_area_error,
                        "Bending Energy": bending_energy,
                        "Stretching Energy": stretching_energy,
                        "Theoretical Bending Energy": theoretical_bending_energy,
                        "Theoretical Stretching Energy": theoretical_stretching_energy,
                        "Percent Error Bending": percent_error_bending,
                        "Percent Error Stretching": percent_error_stretching,
                        "Run Time (s)": run_time,
                        "Perturbed": perturbed_flag
                    })

                    results_df = pd.DataFrame([results[-1]])
                    results_df.to_csv(csv_filename, mode='a', header=not csv_exists, index=False)
                    csv_exists = True



                                    # shape, shape_perturbed = generate_pv_shapes(
                #     shape_name, num_points=num_points, perturbation_strength=perturbation_strength, radius=radius
                # )

                # shape_filename = f"{test_shapes_dir}/{shape_name}_radius_{radius}_points_{num_points}.ply"
                # shape_perturbed_filename = f"{test_shapes_dir}/{shape_name}_radius_{radius}_points_{num_points}_perturbed.ply"

                # save_points_to_ply(np.asarray(shape.points), shape_filename)
                # save_points_to_ply(np.asarray(shape_perturbed.points), shape_perturbed_filename)

                # for variant, filename, perturbed_flag in [("Unperturbed", shape_filename, False), 
                #                                         ("Perturbed", shape_perturbed_filename, True)]:
                #     try:
                #         bending_energy, stretching_energy, computed_area = validate_shape(filename, "N", shape_name, variant, radius)
                #     except Exception as e:
                #         logging.error(f"Error processing {shape_name} ({variant}): {e}")
                #         bending_energy, stretching_energy, computed_area = "Error", "Error", "Error"

                #     computed_area = float(computed_area) if computed_area != "Error" else float('nan')
                #     percent_error = 100 * abs((theoretical_area - computed_area) / theoretical_area) if theoretical_area > 0 else float('nan')

                #     results.append({
                #         "Shape": f"{shape_name}_{variant}",
                #         "Radius": radius,
                #         "Num Points": num_points,
                #         "Point Density": density,
                #         "Theoretical Area": theoretical_area,
                #         "Computed Area": computed_area,
                #         "Percent Error": percent_error,
                #         "Bending Energy": bending_energy,
                #         "Stretching Energy": stretching_energy,
                #         "Theoretical Bending Energy": theoretical_bending_energy,
                #         "Theoretical Stretching Energy": theoretical_stretching_energy,
                #         "Perturbed": perturbed_flag
                #     })

                #     results_df = pd.DataFrame([results[-1]])
                #     results_df.to_csv(csv_filename, mode='a', header=not csv_exists, index=False)
                #     csv_exists = True

    print("Testing completed.")
    pd.DataFrame(results).to_csv("backup_shape_comparison_results.csv", index=False)
