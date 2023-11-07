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
neighbors_for_surface_fit = [5, 10, 15, 20, 25, 30] #minimum of 4!
voxel_sizes = [3, 2, 1, 0.5, 0] #set to zero if you don't need to downsample
# cloud_type = 'sridge' #sridge, sphere, torus, klein, dupin, monkey, bumpy_spheroid
########################################################################



########################################################################
# Don't touch below here unless you wish to modify execution behavior
########################################################################
if __name__ == '__main__':
    for cloud_type in ['sridge', 'sphere', 'torus', 'klein', 'dupin', 'monkey', 'bumpy_spheroid']:
        for voxel_size in voxel_sizes:
            for neighbors in neighbors_for_surface_fit:

                neighbors_for_tree = neighbors

                if voxel_size == 0 or cloud_type != 'sridge':
                    downsample = False
                else:
                    downsample = True

                if cloud_type == 'sridge':
                    print(f'Running for s-ridge with starting neighbors of {neighbors} and voxel size of {voxel_size}')
                    pcl = PointCloud('./sample_scans/sridge.txt', downsample, voxel_size=voxel_size, k_neighbors=neighbors_for_tree, output_path ='./output/sridge/')

                elif cloud_type == 'sphere':
                    print(f'Running for sphere')
                    pcl = PointCloud('./sample_scans/sridge.txt', downsample, voxel_size=voxel_size, k_neighbors=neighbors_for_tree, output_path = './output/sphere/')
                    pcl.generate_sphere_point_cloud(num_points=10000, radius=10)

                elif cloud_type == 'klein':
                    print(f'Running for klein bottle')
                    pcl = PointCloud('./sample_scans/klein_bottle.txt', downsample, voxel_size=voxel_size, k_neighbors=neighbors_for_tree, output_path = './output/klein/')

                elif cloud_type == 'torus':
                    print(f'Running for torus')
                    pcl = PointCloud('./sample_scans/torus.txt', downsample, voxel_size=voxel_size, k_neighbors=neighbors_for_tree, output_path = './output/torus/')

                elif cloud_type == 'monkey':
                    print(f'Running for monkey saddle')
                    pcl = PointCloud('./sample_scans/monkey_saddle.txt', downsample, voxel_size=voxel_size, k_neighbors=neighbors_for_tree, output_path = './output/monkey/')
                
                elif cloud_type == 'dupin':
                    print(f'Running for dupin cyclide')
                    pcl = PointCloud('./sample_scans/dupin_cyclide.txt', downsample, voxel_size=voxel_size, k_neighbors=neighbors_for_tree, output_path = './output/dupin/')
                
                elif cloud_type == 'bumpy_spheroid':
                    print(f'Running for bumpy spheroid')
                    pcl = PointCloud('./sample_scans/bumpy_spheroid.txt', downsample, voxel_size=voxel_size, k_neighbors=neighbors_for_tree, output_path = './output/bumpy_spheroid/')

                else:
                    print('Cloud type not recognized, exiting')
                    exit()



                
                
                # print("denoising the point cloud")
                # pcl.remove_noise_from_point_cloud(k=neighbors, alpha=0.5)

                print("planting tree")
                pcl.plant_kdtree(k_neighbors=neighbors)

                print("plotting knn points")
                pcl.visualize_knn_for_n_random_points(num_points_to_plot=num_visualization_demo_points, k_neighbors=neighbors)

                #####################################################
                # print("Running neighbor study")
                # pcl.implicit_quadric_neighbor_study()

                # print("fitting implicit quadrics")
                # pcl.fit_implicit_quadric_surfaces_all_points()

                # print("calculating quadric curvatures")
                # pcl.calculate_curvatures_of_implicit_quadric_surfaces_for_all_points()

                # print("plotting points colored by curvature")
                # pcl.plot_points_colored_by_quadric_curvatures()
                #####################################################
                print("Running neighbor study")
                pcl.explicit_quadratic_neighbor_study() # use 'goldmans' or 'standard'

                print("Calculating quadratic surfaces")
                pcl.fit_explicit_quadratic_surfaces_to_neighborhoods()

                print("calculating quadratic curvatures")
                pcl.calculate_curvatures_of_explicit_quadratic_surfaces_for_all_points() # use 'goldmans' or 'standard'

                print("plotting quadratic curvatures")
                pcl.plot_points_colored_by_quadratic_curvatures()
                #####################################################

                plt.close('all')
