import open3d as o3d
import os

def downsample_point_cloud(input_ply_file, output_ply_file, voxel_size=0.05):
    # Load the point cloud from the PLY file
    pcd = o3d.io.read_point_cloud(input_ply_file)
    print(f"Original point cloud has {len(pcd.points)} points.")
    
    # Downsample the point cloud using a voxel size
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    print(f"Downsampled point cloud has {len(downsampled_pcd.points)} points.")
    
    # Save the downsampled point cloud to a new PLY file
    o3d.io.write_point_cloud(output_ply_file, downsampled_pcd)
    print(f"Downsampled point cloud saved to {output_ply_file}")

if __name__ == "__main__":
    # Directory containing the input files
    input_dir = "test_shapes"
    output_dir = "test_shapes/downsampled"
    voxel_size = 0.25  # Adjust the voxel size based on the desired downsampling level

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List of all .ply files in the input directory
    ply_files = [
        "1_Scan_POS=196.143.ply",
        "1st_sridge_9-3-2024.ply",
        "1st_unbind_9_3_2024.ply",
        "2_Scan_POS=194.022.ply",
        "2nd_sridge_9_3_2024.ply",
        "2nd_unbind_9-3-2024.ply",
        "3rd_sridge_9_3_2024.ply",
        "3rd_unbind_9_3_2024.ply",
        "4th_sridge_9_5_2024.ply",
        "4th_unbind_9_5_2024.ply",
        "5th_sridge_9_5_2024.ply",
        "5th_unbind_9_5_2024.ply",
        "6th_sridge_9_5_2024.ply",
        "6th_unbind_9_5_2024.ply",
        "7th_sridge_9_5_2024.ply",
        "7th_unbind_9_5_2024.ply",
        "8th_sridge_9_5_2024.ply",
        "8th_unbind_9_5_2024.ply"
    ]

    # Loop over each PLY file and apply the downsampling
    for ply_file in ply_files:
        input_ply = os.path.join(input_dir, ply_file)
        output_ply = os.path.join(output_dir, f"downsampled_{ply_file}")
        downsample_point_cloud(input_ply, output_ply, voxel_size)
