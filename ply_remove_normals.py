from plyfile import PlyData, PlyElement
import numpy as np

def convert_ply_to_xyz(input_file, output_file):
    # Read the input PLY file
    plydata = PlyData.read(input_file)
    
    # Extract x, y, z properties
    vertices = plydata['vertex']
    xyz_data = [(v['x'], v['y'], v['z']) for v in vertices]
    
    # Define the new PLY structure with only x, y, z
    new_vertices = np.array(xyz_data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    new_element = PlyElement.describe(new_vertices, 'vertex')
    
    # Write the new PLY file
    PlyData([new_element], text=True).write(output_file)
    print(f"Converted PLY file saved to {output_file}")

# Usage example
input_ply_file = "Scans/5th_unbind_9_5_2024_CR.ply"  # Replace with the actual input PLY file name
output_ply_file = "no_normals_5th_unbind_9_5_2024_CR.ply"  # Replace with the desired output PLY file name
convert_ply_to_xyz(input_ply_file, output_ply_file)
