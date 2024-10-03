import os
import random
import numpy as np

def read_asc_file(file_path):
    """
    Reads the .asc file, assuming the file has columns in the format: x, y, z, nx, ny, nz (no headers).
    Returns a list of (x, y, z) coordinates after dropping the nx, ny, nz columns.
    """
    coordinates = []
    with open(file_path, 'r') as f:
        for line in f:
            # Split the line into values, and convert them to floats
            values = line.split()
            if len(values) == 6:
                x, y, z = values[:3]  # Take the first three values (x, y, z)
                coordinates.append((float(x), float(y), float(z)))
    return coordinates

def downsample(coordinates, voxel_size=0.1):
    """
    Downsamples the coordinates by voxel grid.
    
    Parameters:
    - coordinates: numpy array of shape (N, 3) where N is the number of points.
    - voxel_size: the size of the voxel grid used for downsampling.
    
    Returns:
    - downsampled_coordinates: numpy array of downsampled points.
    """
    # Convert coordinates to a numpy array if it is not already
    coordinates = np.array(coordinates)
    
    # Compute the voxel indices for each point
    voxel_indices = np.floor(coordinates / voxel_size).astype(np.int32)

    # Create a dictionary to store unique voxels and one point from each voxel
    voxel_dict = {}

    for i, voxel in enumerate(voxel_indices):
        # Use tuple as key because numpy arrays can't be dictionary keys
        voxel_key = tuple(voxel)

        # If the voxel is not already in the dictionary, add the corresponding point
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = coordinates[i]

    # Get all the representative points (one per voxel)
    downsampled_coordinates = np.array(list(voxel_dict.values()))

    return downsampled_coordinates

def write_ply_file(file_path, coordinates):
    """
    Writes a .ply file with the given coordinates and a fixed header.
    """
    with open(file_path, 'w') as f:
        # Write the PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(coordinates)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        # Write the (x, y, z) coordinates
        for coord in coordinates:
            f.write(f"{coord[0]} {coord[1]} {coord[2]}\n")

def convert_asc_to_ply(asc_directory, ply_directory, downsample_fraction=0.1):
    """
    Converts all .asc files in the given directory to .ply files.
    The .ply files will be saved in the specified output directory.
    """
    # Ensure the output directory exists
    if not os.path.exists(ply_directory):
        os.makedirs(ply_directory)

    # Process each .asc file
    for asc_filename in os.listdir(asc_directory):
        if asc_filename.endswith('.asc'):
            asc_file_path = os.path.join(asc_directory, asc_filename)
            ply_filename = os.path.splitext(asc_filename)[0] + '.ply'
            ply_file_path = os.path.join(ply_directory, ply_filename)

            print(f"Converting {asc_filename} to {ply_filename}...")

            # Read the .asc file and extract (x, y, z) coordinates
            coordinates = read_asc_file(asc_file_path)

            # Downsample the coordinates
            sampled_coordinates = downsample(coordinates, downsample_fraction)

            # Write the .ply file with the specified header
            write_ply_file(ply_file_path, sampled_coordinates)

            print(f"Saved: {ply_file_path}")

# Example usage:
asc_directory = './Scans'  # Directory containing .asc files
ply_directory = './test_shapes'  # Directory to save .ply files
voxel_size_for_downsample = 0.1

convert_asc_to_ply(asc_directory, ply_directory, voxel_size_for_downsample)
