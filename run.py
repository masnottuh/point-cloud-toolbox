########################################################################
# Driver used to process point clouds, pulls in pointCloudToolbox.py
# Robert Sam Hutton, 2023
# rhutton@unr.edu or sam@samhutton.net
########################################################################
from pointCloudToolbox import *
########################################################################



########################################################################
# Modify these variables to change the behavior of the program
########################################################################
num_visualization_demo_points = 5
neighbors_for_surface_fit = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 500, 600]
voxel_size = [1, 0.5] #set to zero if you don't need to downsample
cloud_type = 'sridge' #sridge, sphere, kong, or monkey-saddle
########################################################################



########################################################################
# Don't touch below here unless you wish to modify execution behavior
########################################################################
if __name__ == '__main__':
    for voxel_size in voxel_size:
        for neighbor in neighbors_for_surface_fit:
            
            neighbors = neighbor
            neighbors_for_tree = neighbors

            if cloud_type == 'sridge':
                print(f'Running for s-ridge with starting neighbors of {neighbors} and voxel size of {voxel_size}')
                pcl = PointCloud('./sample_scans/sridge.txt', downsample=True, voxel_size=voxel_size, k_neighbors=neighbors_for_tree)

            elif cloud_type == 'sphere':
                pcl = PointCloud('./sample_sacns/sridge.txt', downsample=True, voxel_size=voxel_size, k_neighbors=neighbors_for_tree)
                pcl.generate_sphere_point_cloud(num_points=10000, radius=1)

            elif cloud_type == 'monkey-saddle':
                pcl = PointCloud('./sample_sacns/sridge.txt', downsample=True, voxel_size=voxel_size, k_neighbors=neighbors_for_tree)
                pcl.generate_monkey_saddle_point_cloud(-10.0, 10.0, -10.0, 10.0, 10000)

            elif cloud_type == 'kong':
                pcl = PointCloud('./sample_sacns/kong.txt', downsample=True, voxel_size=voxel_size, k_neighbors=neighbors_for_tree)
            
            # print("denoising the point cloud")
            # pcl.remove_noise_from_point_cloud(k=neighbors, alpha=0.5)

            print("planting tree")
            pcl.plant_kdtree(k_neighbors=neighbors)

            print("plotting knn points")
            pcl.visualize_knn_for_n_random_points(num_points_to_plot=num_visualization_demo_points, k_neighbors=neighbors)

            print("Running neighbor study")
            pcl.quadratic_neighbor_study()

            print("Calculating quadratic surfaces")
            pcl.fit_quadratic_surfaces_to_neighborhoods(neighbors)

            print("calculating quadratic curvatures")
            pcl.calculate_quadratic_curvatures()

            # print("Filtering outliers")
            # pcl.filter_outliers_by_std_dev(num_std_devs=3)

            print("plotting quadratic curvatures")
            pcl.plot_points_colored_by_quadratic_curvatures()

            plt.close('all')
