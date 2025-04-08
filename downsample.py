import open3d as o3d
import os

def downsample_point_cloud(input_ply_file, output_ply_file, voxel_size=1):
    # Load the point cloud from the PLY file
    pcd = o3d.io.read_point_cloud(input_ply_file)
    print(f"Original point cloud has {len(pcd.points)} points.")
    
    # Downsample the point cloud using a voxel size
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    print(f"Downsampled point cloud has {len(downsampled_pcd.points)} points.")
    
    # Save the downsampled point cloud to a new PLY file
    o3d.io.write_point_cloud(output_ply_file, downsampled_pcd, write_ascii=True)

    print(f"Downsampled point cloud saved to {output_ply_file}")

if __name__ == "__main__":
    # Directory containing the input files
    input_dir = "Scans/"
    output_dir = "Scans/"
    voxel_size = 0.4  # Adjust the voxel size based on the desired downsampling level

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List of all .ply files in the input directory
    ply_files = ["NN_u_state5_3_full_C.ply"]

    # Loop over each PLY file and apply the downsampling
    for ply_file in ply_files:
        input_ply = os.path.join(input_dir, ply_file)
        output_ply = os.path.join(output_dir, f"downsampled_{ply_file}")
        downsample_point_cloud(input_ply, output_ply, voxel_size)
