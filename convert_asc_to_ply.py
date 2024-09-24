import os

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

def convert_asc_to_ply(asc_directory, ply_directory):
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

            # Write the .ply file with the specified header
            write_ply_file(ply_file_path, coordinates)

            print(f"Saved: {ply_file_path}")

# Example usage:
asc_directory = './Scans'  # Directory containing .asc files
ply_directory = './test_shapes'  # Directory to save .ply files

convert_asc_to_ply(asc_directory, ply_directory)
