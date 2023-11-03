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
neighbors_for_surface_fit = [5,9,15]
smooth_using_moving_mean, neighbors_for_moving_mean = True, 3
voxel_size = [0.5, 0.25, 0.1] #set to zero if you don't need to downsample
use_feature_fitting = False #Currently Broken
# smallest_characteristic_length = 1/10 # approximate ratio of the smallest feature to the size of the point cloud

rotation_angle_x = 0 #rotate the point cloud if needed
rotation_angle_y = 0
rotation_angle_z = 0
########################################################################



########################################################################
# Don't touch below here unless you wish to modify execution behavior
########################################################################
if __name__ == '__main__':
    for voxel_size in voxel_size:
        for neighbor in neighbors_for_surface_fit:
            
            neighbors = neighbor
            neighbors_for_tree = neighbors

            print(f'Running with starting neighbors of {neighbors} and voxel size of {voxel_size}')

            pcl = PointCloud('./sridge.txt', downsample=True, voxel_size=voxel_size, k_neighbors=neighbors_for_tree)
            pcl.remove_noise_from_point_cloud(k=neighbors, alpha=0.5)
            # pcl.generate_sphere_point_cloud(radius=10, num_points=10000)
            # pcl.plot_surface()
 
            # rotation_rad_x = np.pi*rotation_angle_x/180 
            # rotation_rad_y = np.pi*rotation_angle_y/180
            # rotation_rad_z = np.pi*rotation_angle_z/180
            # pcl.rotate_point_cloud(rotation_angle_x=rotation_rad_x, rotation_angle_y=rotation_rad_y, rotation_angle_z=rotation_rad_z)
            # pcl.plot_surface()
            print("planting tree")
            pcl.plant_kdtree(k_neighbors=neighbors)

            # if smooth_using_moving_mean:
            #     pcl.smooth_point_cloud_by_neighborhood_moving_mean(k_neighbors=neighbors) 

            print("plotting knn points")
            pcl.visualize_knn_for_n_random_points(num_points_to_plot=num_visualization_demo_points, k_neighbors=neighbors)

            # print("fitting quadratic surfaces")
            # pcl.fit_quadratic_surfaces_to_neighborhoods(k_neighbors=neighbors)
            # print("calculating parametric curvatures")
            # pcl.calculate_parametric_curvatures_direct()
            # print("rejecting outliers")
            # pcl.reject_outliers_curvature()
            # print("plotting parametric curvaturess")
            # pcl.plot_parametric_curvatures()
            
            # pcl.find_optimal_num_neighbors()
            pcl.fit_quadric_surfaces()
            # pcl.plot_quadric_surfaces()
            pcl.calculate_quadric_curvatures()
            # pcl.filter_outlier_curvatures_per_neighborhood(threshold_std_devs=3)
            pcl.plot_points_colored_by_quadric_curvatures()
            
            # print("calculating pseudo curvatures")
            # pcl.calculate_pseudo_parametric_curvatures()
            # print("plotting pseudo curvatures")
            # pcl.plot_pseudo_parametric_curvatures()
            # curvature by principal curvature analysis
            # print("calculating principal curvatures")
            # pcl.principal_curvatures_via_principal_component_analysis(k_neighbors=neighbors)
            # print("plotting principal curvatures")
            # pcl.plot_principal_curvatures_from_principal_component_analysis()
            # print("plotting mean and gaussian curvatures")
            # pcl.plot_mean_and_gaussian_curvatures_from_principal_component_analysis()
            # pcl.plot_principal_curvature_directions_from_principal_component_analysis()

            plt.close('all')
