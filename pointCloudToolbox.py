########################################################################
# Class used to process point clouds
# Robert "Sam" Hutton 
# rhutton@unr.edu or sam@samhutton.net
########################################################################
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy as sp
from scipy.linalg import eigh
import sympy as sympy
from scipy.optimize import least_squares
import pickle
import random
from mayavi import mlab
from sympy.plotting import plot3d_parametric_surface

class PointCloud:

    def __init__(self, file_path, downsample, voxel_size, k_neighbors, max_points_per_voxel=1):
        #initial parameters
        self.file_path = file_path
        self.downsample = downsample
        self.k_neighbors = k_neighbors
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.readFromFile()

        #secondary parameters
        self.num_points = len(self.points)
        self.num_features = len(self.points[0])
        self.l1_norm = np.linalg.norm(self.points, 1)
        self.l2_norm = np.linalg.norm(self.points, 2)
        self.infinity_norm = np.linalg.norm(self.points, np.inf)

    @staticmethod
    def running_mean_vector(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    
    @staticmethod
    def running_mean_outlier(x, N):

        delta_vector = np.zeros(len(x))

        for i, item in enumerate(x):
            if i == 0:
                delta_vector[i] = 0
            else:
                delta_vector[i] = np.abs(x[i] - x[i-1])

        delta_mean = np.average(delta_vector)
        delta_std = np.std(delta_vector)

        for i, item in enumerate(x):
            if i < N:
                average = np.average(x[:i+N])
                if delta_vector[i] > delta_mean + 2*delta_std or delta_vector[i] < delta_mean - 2*delta_std:
                    x[i] = average

            elif i > len(x)-N:
                average = np.average(x[i-N:])
                if delta_vector[i] > delta_mean + 2*delta_std or delta_vector[i] < delta_mean - 2*delta_std:
                    x[i] = average

            else:
                average = np.average(x[i-N:i+N])
                if delta_vector[i] > delta_mean + 2*delta_std or delta_vector[i] < delta_mean - 2*delta_std:
                    x[i] = average

            cumsum = np.cumsum(np.insert(x, 0, 0)) 
            return (cumsum[N:] - cumsum[:-N]) / float(N)

    @staticmethod
    def filter_outliers_median(data, threshold=2.6):
        data = np.array(data)
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        window_size = 1

        # Define the threshold for outliers
        threshold_value = threshold * mad
        
        # Create a boolean mask to identify outliers
        is_outlier = np.abs(data - median) > threshold_value
        
        # Create a rolling window view of the data
        rolled_data = np.lib.stride_tricks.sliding_window_view(data, window_shape=window_size)
        
        # Calculate the mean of neighbors for each window
        neighbor_means = np.mean(rolled_data, axis=1)
        
        # Replace outliers with the mean of their neighbors
        data[is_outlier] = neighbor_means[is_outlier]
        
        return data.tolist()

    @staticmethod
    def filter_outliers_absolute(data, max_abs=0.01):
        data = np.array(data)
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        window_size = 1
        
        # Create a boolean mask to identify outliers
        is_outlier = np.abs(data) > max_abs
        
        # Create a rolling window view of the data
        rolled_data = np.lib.stride_tricks.sliding_window_view(data, window_shape=window_size)
        
        # Replace outliers with the mean of their neighbors
        data[is_outlier] = np.nan
        
        return data.tolist()

    def readFromFile(self):

        points = np.loadtxt(self.file_path)
        # print("Points and normals loaded from file.")

        self.points = points[:, 0:3]
        self.normals = points[:, 3:6]

        #shift the points to the positive quadrant
        self.points[:, 0] = self.points[:, 0] - np.max(self.points[:, 0])
        self.points[:, 1] = self.points[:, 1] - np.max(self.points[:, 1])

        #downsample if set to true, otherwise just use the points as is
        if self.downsample==True:
            self.points, self.normals = self.downsample_point_cloud_by_grid()
            #shift the points to the positive quadrant
            self.points[:, 0] = self.points[:, 0] - np.min(self.points[:, 0])
            self.points[:, 1] = self.points[:, 1] - np.min(self.points[:, 1])
            # print(f"Point cloud downsampled using voxel size of {self.voxel_size} and max points per voxel of {self.max_points_per_voxel}.")

        self.x_domain = [np.min(self.points[:, 0]), np.max(self.points[:, 0])]
        self.y_domain = [np.min(self.points[:, 1]), np.max(self.points[:, 1])]
        self.z_domain = [np.min(self.points[:, 2]), np.max(self.points[:, 2])]

    def uniform_grid_resampling(self):
        point_cloud = self.points
        normals = self.normals
        grid_spacing = self.voxel_size
        max_points_per_voxel = self.max_points_per_voxel
        # Calculate the minimum and maximum coordinates (bounding box) of the point cloud.
        min_coords = np.min(point_cloud, axis=0)
        max_coords = np.max(point_cloud, axis=0)

        # Create a 3D grid with the specified grid spacing.
        grid_x = np.arange(min_coords[0], max_coords[0], grid_spacing)
        grid_y = np.arange(min_coords[1], max_coords[1], grid_spacing)
        grid_z = np.arange(min_coords[2], max_coords[2], grid_spacing)
        
        # Create a grid of cell centers in 3D space.
        grid_centers = np.array(np.meshgrid(grid_x, grid_y, grid_z)).T.reshape(-1, 3)

        # Initialize arrays to store resampled points and normals.
        resampled_points = []
        resampled_normals = []

        # Build a k-d tree for efficient nearest-neighbor search.
        kdtree = cKDTree(point_cloud)

        # Iterate through each grid cell and find the nearest point.
        for grid_center in grid_centers:
            # Query the nearest point in the original point cloud.
            _, nearest_point_index = kdtree.query(grid_center)
            
            # Append the nearest point and its corresponding normal.
            resampled_points.append(point_cloud[nearest_point_index])
            resampled_normals.append(normals[nearest_point_index])

        # Convert the resampled points and normals to numpy arrays.
        resampled_points = np.array(resampled_points)
        resampled_normals = np.array(resampled_normals)

        return resampled_points, resampled_normals

    def voxel_grid_downsampling(self):
        point_cloud = self.points
        normals = self.normals
        voxel_size = self.voxel_size
        max_points_per_voxel = self.max_points_per_voxel

        # Calculate voxel indices for each point
        voxel_indices = np.floor(point_cloud / voxel_size).astype(int)

        # Calculate the voxel centers
        voxel_centers = (voxel_indices + 0.5) * voxel_size

        # Find unique voxel indices to select one point per voxel
        unique_voxel_indices, voxel_counts = np.unique(voxel_indices, return_counts=True)

        # Initialize arrays to store the downsampled point cloud
        downsampled_points = np.zeros((len(unique_voxel_indices), 3))
        downsampled_normals = np.zeros((len(unique_voxel_indices), 3))

        for i, voxel_index in enumerate(unique_voxel_indices):
            # Find all points in the current voxel
            points_in_voxel = point_cloud[(voxel_indices == voxel_index).all(axis=1)]

            if len(points_in_voxel) > 0:
                # Calculate distances from voxel center to all points in the voxel
                distances = np.linalg.norm(points_in_voxel - voxel_centers[i], axis=1)

                # Find the index of the closest point
                closest_point_index = np.argmin(distances)

                # Store the closest point and its normal in the downsampled arrays
                downsampled_points[i] = points_in_voxel[closest_point_index]
                downsampled_normals[i] = normals[(voxel_indices == voxel_index).all(axis=1)][closest_point_index]

        return downsampled_points, downsampled_normals
            
    def downsample_point_cloud_uniformly(self):

        kdtree = cKDTree(self.points)
        points = self.points
        normals = self.normals
        voxel_size = self.voxel_size
        max_points_per_voxel = self.max_points_per_voxel


        # Initialize lists to store selected points and normals
        selected_points = []
        selected_normals = []

        # Calculate the voxel grid based on voxel_size
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        num_voxels = np.ceil((max_bounds - min_bounds) / voxel_size).astype(int)
        
        for x in range(num_voxels[0]):
            for y in range(num_voxels[1]):
                for z in range(num_voxels[2]):
                    voxel_center = min_bounds + np.array([x, y, z]) * voxel_size

                    # Find all points within a certain radius (Poisson disk radius)
                    search_radius = voxel_size / 2.0
                    nearby_point_indices = kdtree.query_ball_point(voxel_center, search_radius)

                    if nearby_point_indices:
                        # Randomly select points from the nearby points
                        random.shuffle(nearby_point_indices)
                        num_points_to_select = min(len(nearby_point_indices), max_points_per_voxel)
                        selected_points.extend(points[nearby_point_indices[:num_points_to_select]])
                        selected_normals.extend(normals[nearby_point_indices[:num_points_to_select]])

        return np.array(selected_points), np.array(selected_normals)
    
    def downsample_point_cloud_by_grid_DEPRECATED(self, max_points_per_voxel=1):
        point_cloud = self.points
        normals = self.normals
        voxel_size = self.voxel_size

        # Extract x, y, z coordinates from the input point cloud
        xyz = point_cloud[:, :3]
        
        # Calculate the minimum and maximum bounds of the point cloud
        min_bound = np.min(xyz, axis=0)
        max_bound = np.max(xyz, axis=0)
        
        # Calculate the number of voxels in each dimension
        num_voxels = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
        
        # Calculate the indices of the voxels for each point
        voxel_indices = ((xyz - min_bound) / voxel_size).astype(int)
        
        # Initialize a dictionary to store the indices of points in each voxel
        voxel_point_indices = {}
        
        # Group point indices into voxels
        for i, indices in enumerate(voxel_indices):
            voxel_key = tuple(indices)
            if voxel_key not in voxel_point_indices:
                voxel_point_indices[voxel_key] = []
            voxel_point_indices[voxel_key].append(i)
        
        # Select points from each voxel with up to 'max_points_per_voxel' points
        selected_indices = []
        for indices in voxel_point_indices.values():
            selected_indices.extend(indices[:max_points_per_voxel])
        
        # Extract the selected points
        ddownsampled_point_cloud = point_cloud[selected_indices]
        dnormals = normals[selected_indices]
        
        return ddownsampled_point_cloud, dnormals

    def downsample_point_cloud_by_grid(self):
        # Calculate the minimum and maximum coordinates of the point cloud
        xyz = self.points[:, :3]
        min_bounds = np.min(xyz, axis=0)
        max_bounds = np.max(xyz, axis=0)
        max_points_per_voxel=1

        # Calculate the number of voxels in each dimension
        num_voxels = np.ceil((max_bounds - min_bounds) / self.voxel_size).astype(int)

        # Calculate the indices of the voxels for each point
        voxel_indices = ((self.points[:, :3] - min_bounds) / self.voxel_size).astype(int)

        # Initialize a dictionary to store the indices of points in each voxel
        voxel_point_indices = {}

        # Group point indices into voxels
        for i, indices in enumerate(voxel_indices):
            voxel_key = tuple(indices)
            if voxel_key not in voxel_point_indices:
                voxel_point_indices[voxel_key] = []
            voxel_point_indices[voxel_key].append(i)

        # Select points from each voxel with up to 'max_points_per_voxel' points
        selected_indices = []
        for indices in voxel_point_indices.values():
            selected_indices.extend(indices[:max_points_per_voxel])

        # Extract the selected points
        downsampled_points = self.points[selected_indices]
        downsampled_normals = self.normals[selected_indices]
        

        # Update the class attribute 'points' with the downsampled points
        return downsampled_points, downsampled_normals

    def plot_surface(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c='b', s=1.5)
        ax.set_title('Point Cloud')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        pickle.dump(fig, open(f'point_cloud_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))
        # plt.show()
        # ax.set_xlim3d(-max(self.points[:,0]), max(self.points[:,0]))
        # ax.set_ylim3d(-max(self.points[:,1]), max(self.points[:,1]))
        # plt.show()

    def rotate_point_cloud(self, rotation_angle_x, rotation_angle_y, rotation_angle_z):

        # Rotate the point cloud to be in the x-y plane
        points = np.column_stack((self.points[:, 1], self.points[:, 2], self.points[:, 0]))
        # Sort the point cloud by x coordinate
        sorted_indices = np.lexsort((points[:, 0], points[:, 1]))
        points = points[sorted_indices]

        # Center the point cloud
        center = np.mean(self.points, axis=0)
        centered = self.points - center
        self.points = centered

        # Create rotation matrices
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, np.cos(rotation_angle_x), -np.sin(rotation_angle_x)],
            [0, np.sin(rotation_angle_x), np.cos(rotation_angle_x)]
        ])

        rotation_matrix_y = np.array([
            [np.cos(rotation_angle_y), 0, np.sin(rotation_angle_y)],
            [0, 1, 0],
            [-np.sin(rotation_angle_y), 0, np.cos(rotation_angle_y)]
        ])

        rotation_matrix_z = np.array([
            [np.cos(rotation_angle_y), -np.sin(rotation_angle_y), 0],
            [np.sin(rotation_angle_y), np.cos(rotation_angle_y), 0],
            [0, 0, 1]
        ])

        # Apply the rotations
        self.points = centered.dot(rotation_matrix_x).dot(rotation_matrix_y).dot(rotation_matrix_z)
        self.points = self.points + center

    def plant_kdtree(self, k_neighbors):
        self.k_neighbors = k_neighbors
        box_size = [int(np.ceil(self.voxel_size*self.k_neighbors)/2), int(np.ceil(self.voxel_size*self.k_neighbors)/2)]

        # self.kdtree = sp.spatial.cKDTree(self.points, box_size) #to use box size for even distribution, CURRENTLY BROKEN
        self.kdtree = sp.spatial.cKDTree(np.array(self.points))

        self.dists = []
        self.neighbor_indices = []
        for i, point in enumerate(self.points):
            dists, neighbor_indices = self.kdtree.query(np.array(point), k_neighbors+1) # +1 because will also return self
            self.dists.append(dists[1:])
            self.neighbor_indices.append(neighbor_indices[1:])
        

        # self.dists, self.neighbor_indices = self.kdtree.query(self.points, k_neighbors)
        # count_neighbors(self, other, r[, p, ...])
        # Count how many nearby pairs can be formed.

        # query(self, x[, k, eps, p, ...])
        # Query the kd-tree for nearest neighbors

        # query_ball_point(self, x, r[, p, eps, ...])
        # Find all points within distance r of point(s) x.

        # query_ball_tree(self, other, r[, p, eps])
        # Find all pairs of points between self and other whose distance is at most r

        # query_pairs(self, r[, p, eps, output_type])
        # Find all pairs of points in self whose distance is at most r.

        # sparse_distance_matrix(self, other, max_distance)
        # Compute a sparse distance matrix

    def smooth_point_cloud_by_neighborhood_moving_mean(self, k_neighbors):
        smoothed_point_cloud = np.zeros_like(self.points)

        for i, point in enumerate(self.points):
            
            smoothed_point = [0, 0, 0]
            neighbors = self.points[self.neighbor_indices[i]]
            
            smoothed_point[0] = point[0]
            smoothed_point[1] = point[1]
            smoothed_point[2] = np.average(neighbors[:, 2])
            smoothed_point_cloud[i] = smoothed_point

        self.points = smoothed_point_cloud
        self.plant_kdtree(k_neighbors=self.k_neighbors)

    def visualize_knn_for_n_random_points(self, num_points_to_plot, k_neighbors):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_title(f'K Nearest Neighbors  K = {k_neighbors}, Voxel Size = {self.voxel_size}')

        self.random_indexes = np.random.randint(0, len(self.points), num_points_to_plot)
        given_points = self.points[self.random_indexes]

        # print(f"Plotting neighbors for {num_points_to_plot} random points in the point cloud.")

        self.random_points = given_points

        for i, index in enumerate(self.random_indexes):
            point = self.points[index]
            neighbors = self.points[self.neighbor_indices[index]]
            # print(neighbors)
            ax.scatter(point[0], point[1], point[2], c='b', s=1)
            ax.scatter(neighbors[:, 0], neighbors[: , 1], neighbors[: ,2], c='r', s=1)

        pickle.dump(fig, open(f'nearest_neighbors_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))
        # plt.show()

    def plot_quadric_surfaces(self):

        for i, index in enumerate(self.random_indexes):
            point = self.points[index]
            A, B, C, D, E, F, G, H, I, J = self.quadric_coefficients[i]
            mlab.clf()
            x, y, z = np.mgrid[-1000:1000:50j, -1000:1000:50j, -1000:1000:50j]
            f = A*x*x+B*y*y+C*z*z+D*x*y+E*x*z+F*y*z+G*x+H*y+I*z+J
            # zed = -(2*A + J + G*x + 2*B*y + H*y + D*x*y)/(2*C + I + E*x + F*y)
            # yeh = -(2 *A + J + G *x + 2 *C *z + I *z + E *x *z)/(2 *B + H + D* x + F *z)
            # ecks = -(2 *A + J + 2 *B *y + H *y + 2* C *z + I* z + F* y* z)/(G + D* y + E *z)
            mlab.points3d(point[0], point[1], point[2], scale_factor=1)
            mlab.figure()
            mlab.contour3d(x, y, z, f)
            mlab.axes()
            mlab.show()
            # plot3d_parametric_surface(ecks, yeh, zed, (x, -5, 5), (y, -5, 5))

    def plot_quadratic_surfaces_fitted(self):

        # print(f'Plotting quadratic surfaces for {len(points_to_plot)} points.')

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
        ax1.set_title(f'Quadratic Surfaces Fitted to Neighborhoods  K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')

        for i, index in enumerate(self.random_indexes):

            point = self.points[index]
            neighbors = self.points[self.neighbor_indices[index]]
            ax1.scatter(point[0], point[1], point[2], c='b', s=1)
            ax1.scatter(neighbors[:, 0], neighbors[: , 1], neighbors[: ,2], c='r', s=1)

        for i, index in enumerate(self.random_indexes):

            ax1.plot_surface(self.X[index], self.Y[index], self.Z[index], rstride=1, cstride=1, alpha=0.5)
    

        ax1.set_zlabel('Z')
        ax1.set_ylabel('Y')
        ax1.set_xlabel('X')

        pickle.dump(fig1, open(f'quadratic_surfaces_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))
            
        # plt.show()
    
    def find_optimal_num_neighbors(self):
        k_neighbors = 2 
        tree = self.kdtree
        points = self.points
        normals = self.normals
        random_indexes = self.random_indexes
        random_points = self.random_points

        test_results = {}

        for i, point in enumerate(random_points):
            test_results[point[0]] = {}
            test_results[point[0]]['gaussian'] = []
            test_results[point[0]]['mean'] = []
            test_results[point[0]]['mean squared'] = []
            test_results[point[0]]['principal_1'] = []
            test_results[point[0]]['principal_2'] = []
            test_results[point[0]]['neighbors'] = []

            for num_neighbors in range(2, int(len(points)/1000)):

                dists, neighbor_indices = tree.query(np.array(point), num_neighbors+1)
                
                neighbors = points[neighbor_indices[1:]]

                min_x, max_x = point[0]-5*self.voxel_size, point[0]+5*self.voxel_size
                min_y, max_y = point[1]-5*self.voxel_size, point[1]+5*self.voxel_size
                min_z, max_z = point[2]-5*self.voxel_size, point[2]+5*self.voxel_size
                
                X,Y,Z = np.meshgrid(np.arange(min_x, max_x, 0.5)), np.meshgrid(np.arange(min_y, max_y, 0.5)), np.meshgrid(np.arange(min_z, max_z, 0.5))

                def quadric_surface(params, points):
                    A, B, C, D, E, F, G, H, I, J = params
                    x, y, z = points.T
                    return A * x ** 2 + B * y ** 2 + C * z ** 2 + D * x * y + E * x * z + F * y * z + G * x + H * y + I * z + J

                def error(params, points):
                    return quadric_surface(params, points)

                initial_params = np.ones(10)  # Initial guess for coefficients

                result = least_squares(error, initial_params, args=(neighbors,))
                optimized_params = result.x

                A, B, C, D, E, F, G, H, I, J = optimized_params

                #calculate the partial derivatives of the surface Ax2+By2+Cz2+Dxy+Exz+Fyz+Gx+Hy+Jz+K=0.
                fx = 2*A*point[0] + D*point[1] + E*point[2] + G
                fy = 2*B*point[1] + D*point[0] + F*point[2] + H
                fz = 2*C*point[2] + E*point[0] + F*point[1] + I
                fxx = 2*A
                fyy = 2*B
                fzz = 2*C
                fxy = D
                fxz = E
                fyz = F
                fyx = D
                fzx = E
                fzy = F

                g = gradient = np.array([fx, fy, fz])
                h = hessian = np.array([[fxx, fxy, 0], [fxy, fyy, 0], [0, 0, fzz]])
                adjoint_of_hessian = adj_h = np.array([[fyy*fzz-fyz*fzy, fyz*fzx-fyx*fzz, fxy*fzy-fyy*fzx], [fxz*fzy-fyx*fzz, fxx*fzz-fxz*fzx, fxy*fzx-fxx*fzy], [fxy*fyz-fxz*fyy, fyx*fxz-fxx*fyz, fxx*fyy-fxy*fyx]])

                # Gaussian Curvature
                K_g = (np.inner(np.inner(g,adj_h),g.T))/(np.linalg.norm(g)**4)

                # Mean Curvature
                K_m = (np.inner(np.inner(g,h),g.T)-np.linalg.norm(g)**2*np.trace(h))/(2*np.linalg.norm(g)**3)
                K_m_squared = K_m**2

                # Principal Curvatures
                k1 = K_m + np.sqrt(K_m**2 - K_g)
                k2 = K_m - np.sqrt(K_m**2 - K_g)


                test_results[point[0]]['neighbors'].append(num_neighbors)
                test_results[point[0]]['gaussian'].append(K_g)
                test_results[point[0]]['mean'].append(K_m)
                test_results[point[0]]['mean squared'].append(K_m_squared)
                test_results[point[0]]['principal_1'].append(k1)
                test_results[point[0]]['principal_2'].append(k2)

        for i, point in enumerate(random_points):
            #plot test results
            figy = plt.figure()
            axx = figy.add_subplot(1, 1, 1)
            axx.set_title(f'Neighbor Test, Voxel Size = {self.voxel_size}')
            axx.set_xlabel('num neighbors')
            axx.set_ylabel('Gaussian Curvature')
            axx.scatter(test_results[point[0]]['neighbors'],test_results[point[0]]['gaussian'], c='b', s=1)
            
            pickle.dump(figy, open(f'Gaussian Curvature {i}.pickle', 'wb'))

            figz = plt.figure()
            axxa = figz.add_subplot(1, 1, 1)
            axxa.set_title(f'Neighbor Test, Voxel Size = {self.voxel_size}')
            axxa.set_xlabel('num neighbors')
            axxa.set_ylabel('Mean Curvature ^ 2')
            axxa.scatter(test_results[point[0]]['neighbors'],test_results[point[0]]['mean squared'], c='b', s=1)
            
            pickle.dump(figz, open(f'Mean Curvature Squared {i}.pickle', 'wb'))

    def plot_points_colored_by_quadric_curvatures(self):
        n = 1
        self.quadric_gaussian_curvatures = self.filter_outliers_absolute(self.quadric_gaussian_curvatures)
        self.quadric_mean_curvatures = self.filter_outliers_absolute(self.quadric_mean_curvatures)
        # Gaussian Curvature from quadric calculations
        fig_curvature_K = plt.figure()
        ax_curvature_K = fig_curvature_K.add_subplot(111, projection='3d')
        sc = ax_curvature_K.scatter(self.points[n-1:, 0], self.points[n-1:, 1], self.points[n-1:, 2], c=self.quadric_gaussian_curvatures, cmap='viridis', s=1)
        fig_curvature_K.colorbar(sc, ax=ax_curvature_K)
        plt.tight_layout()
        ax_curvature_K.view_init(azim=90, elev=85)
        ax_curvature_K.set_axis_off()
        ax_curvature_K.set_title(f'Gaussian Curvature from quadric surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')
        pickle.dump(fig_curvature_K, open(f'Gaussian Curvature from quadric surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}.pickle', 'wb'))

        # Mean Curvature from quadric surface
        fig_curvature_H = plt.figure()
        ax_curvature_H = fig_curvature_H.add_subplot(111, projection='3d')
        sc = ax_curvature_H.scatter(self.points[n-1:, 0], self.points[n-1:, 1], self.points[n-1:, 2], c=self.quadric_mean_curvatures, cmap='viridis', s=1)
        fig_curvature_H.colorbar(sc, ax=ax_curvature_H)
        plt.tight_layout()
        ax_curvature_H.view_init(azim=90, elev=85)
        ax_curvature_H.set_axis_off()
        ax_curvature_H.set_title(f'Mean Curvature from quadric surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')
        pickle.dump(fig_curvature_H, open(f'Mean Curvature from quadric surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}.pickle', 'wb'))

        # plt.show()

        # Plot histograms
        fig_hist_K_fund = plt.figure()
        plt.hist(np.array(self.quadric_gaussian_curvatures, dtype=float), bins=100)
        plt.title(f'Hist Gaussian Curvature from quadric surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')
        plt.tight_layout()
        pickle.dump(fig_hist_K_fund, open(f'Hist Gaussian Curvature from quadric surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}.pickle', 'wb'))

        fig_hist_H_fund = plt.figure()
        plt.hist(np.array(self.quadric_mean_curvatures, dtype=float), bins=100)
        plt.title(f'Hist Mean Curvature from quadric surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')
        plt.tight_layout()
        pickle.dump(fig_hist_H_fund, open(f'Hist Mean Curvature from quadric surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}.pickle', 'wb'))

        # plt.show()

    def calculate_quadric_curvatures(self):
        self.quadric_gaussian_curvatures = [[] for i in range(len(self.points))]
        self.quadric_mean_curvatures = [[] for i in range(len(self.points))]
        self.principal_curvature_1 = [[] for i in range(len(self.points))]
        self.principal_curvature_2 = [[] for i in range(len(self.points))]

        #see R. Goldman / Computer Aided Geometric Design 22 (2005) 632–658 for formulas used
        for i, point in enumerate(self.points):

            A, B, C, D, E, F, G, H, I, J = self.quadric_coefficients[i]
            surface = f = A*(self.x_quadric[i]**2) + B*(self.y_quadric[i]**2) + C*(self.z_quadric[i]**2) + D*(self.x_quadric[i]*self.y_quadric[i]) + E*(self.x_quadric[i]*self.z_quadric[i]) + F*(self.y_quadric[i]*self.z_quadric[i]) + G*(self.x_quadric[i]) + H*(self.y_quadric[i]) + I*(self.z_quadric[i]) + J

            #calculate the partial derivatives of the surface Ax2+By2+Cz2+Dxy+Exz+Fyz+Gx+Hy+Jz+K=0.
            fx = 2*A*self.x_quadric[i] + D*self.y_quadric[i] + E*self.z_quadric[i] + G
            fy = 2*B*self.y_quadric[i] + D*self.x_quadric[i] + F*self.z_quadric[i] + H
            fz = 2*C*self.z_quadric[i] + E*self.x_quadric[i] + F*self.y_quadric[i] + I
            fxx = 2*A
            fyy = 2*B
            fzz = 2*C
            fxy = D
            fxz = E
            fyz = F
            fyx = D
            fzx = E
            fzy = F

            g = gradient = np.array([fx, fy, fz])
            h = hessian = np.array([[fxx, fxy, 0], [fxy, fyy, 0], [0, 0, fzz]])
            adjoint_of_hessian = adj_h = np.array([[fyy*fzz-fyz*fzy, fyz*fzx-fyx*fzz, fxy*fzy-fyy*fzx], [fxz*fzy-fyx*fzz, fxx*fzz-fxz*fzx, fxy*fzx-fxx*fzy], [fxy*fyz-fxz*fyy, fyx*fxz-fxx*fyz, fxx*fyy-fxy*fyx]])

            # Gaussian Curvature
            K_g = (np.inner(np.inner(g,adj_h),g.T))/(np.linalg.norm(g)**4)

            # Mean Curvature
            K_m = (np.inner(np.inner(g,h),g.T)-np.linalg.norm(g)**2*np.trace(h))/(2*np.linalg.norm(g)**3)
            K_m = K_m**2
            # print(K_m)

            # Principal Curvatures
            k1 = K_m + np.sqrt(K_m**2 - K_g)
            k2 = K_m - np.sqrt(K_m**2 - K_g)
            # print(k1, k2)

            # append to lists
            self.quadric_gaussian_curvatures[i] = K_g
            self.quadric_mean_curvatures[i] = K_m
            self.principal_curvature_1[i] = k1
            self.principal_curvature_2[i] = k2

    def fit_quadric_surfaces(self):
        # A is a matrix containing the terms corresponding to the coefficients of the quadric polynomial.
        # x is a vector containing the coefficients [A, B, C, D, E, F, G, H, I, J].
        # b is a vector containing the z-coordinates of the data points.
        # Your 3D point cloud data

        self.quadric_coefficients = [[] for i in range(len(self.points))]
        self.x_quadric = [[] for i in range(len(self.points))]
        self.y_quadric = [[] for i in range(len(self.points))]
        self.z_quadric = [[] for i in range(len(self.points))]
        self.xx_quadric = [[] for i in range(len(self.points))]
        self.yy_quadric = [[] for i in range(len(self.points))]
        self.zz_quadric = [[] for i in range(len(self.points))]


        for i, point in enumerate(self.points):
            
            #grid
            min_x, max_x = point[0]-5*self.voxel_size, point[0]+5*self.voxel_size
            min_y, max_y = point[1]-5*self.voxel_size, point[1]+5*self.voxel_size
            min_z, max_z = point[2]-5*self.voxel_size, point[2]+5*self.voxel_size
            X,Y,Z = np.meshgrid(np.arange(min_x, max_x, 0.5)), np.meshgrid(np.arange(min_y, max_y, 0.5)), np.meshgrid(np.arange(min_z, max_z, 0.5))
            # points = self.points[self.neighbor_indices[i]]
            error = 0.0
            # if 'k' in locals():
            #     print(k)
            # print("successful fit")

            k = self.k_neighbors
            # char_dim = (max(self.points[:,0]-min(self.points[:,0]))/10)
            # while error < 0.00001 and k < char_dim:

            dists, neighbor_indicies = self.kdtree.query(point, k)
            neighbors = self.points[neighbor_indicies]

            # center the points to the origin
            neighbors = neighbors - point

            #points
            # x = np.array(neighbors[:, 0])  # x coordinates
            # y = np.array(neighbors[:, 1])  # y coordinates
            # z = np.array(neighbors[:, 2])  # z coordinates


            def quadric_surface(params, points):
                A, B, C, D, E, F, G, H, I, J = params
                x, y, z = points.T
                return A * x ** 2 + B * y ** 2 + C * z ** 2 + D * x * y + E * x * z + F * y * z + G * x + H * y + I * z + J

            def error(params, points):
                return quadric_surface(params, points)

            initial_params = np.ones(10)  # Initial guess for coefficients

            result = least_squares(error, initial_params, args=(neighbors,))
            optimized_params = result.x

            def quadric_surface_function(x, y, z):
                A, B, C, D, E, F, G, H, I, J = optimized_params
                return A * x ** 2 + B * y ** 2 + C * z ** 2 + D * x * y + E * x * z + F * y * z + G * x + H * y + I * z + J

            error = np.sum(abs(error(optimized_params, neighbors)))
            # print(error)

            # k = k+1
        
            self.quadric_coefficients[i] = optimized_params

            self.x_quadric[i] = 0
            self.y_quadric[i] = 0
            self.z_quadric[i] = 0

    def fit_quadratic_surfaces_to_neighborhoods(self, k_neighbors):

        self.C = [[] for i in range(len(self.points))]
        self.X = [[] for i in range(len(self.points))]
        self.Y = [[] for i in range(len(self.points))]
        self.XX = [[] for i in range(len(self.points))]
        self.YY = [[] for i in range(len(self.points))]
        self.Z = [[] for i in range(len(self.points))]
        
        for i, point in enumerate(self.points):

            points = self.points[self.neighbor_indices[i]]

            min_x, max_x = point[0]-2*self.voxel_size, point[0]+2*self.voxel_size
            min_y, max_y = point[1]-2*self.voxel_size, point[1]+2*self.voxel_size
            X,Y = np.meshgrid(np.arange(min_x, max_x, 0.5), np.arange(min_y, max_y, 0.5))
            XX = X.flatten()
            YY = Y.flatten()

            A = np.c_[np.ones(points.shape[0]), points[:,:2], np.prod(points[:,:2], axis=1), points[:,:2]**2]
            coefficients,_,_,_ = sp.linalg.lstsq(A, points[:,2])

            Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], coefficients).reshape(X.shape)

            self.C[i] = coefficients
            self.X[i] = X
            self.Y[i] = Y
            self.XX[i] = XX
            self.YY[i] = YY
            self.Z[i] = Z

        self.plot_quadratic_surfaces_fitted(self.random_points)

    def reject_outliers_curvature(self):
        self.K_from_components_of_fundamental_forms = self.filter_outliers_median(self.K_from_components_of_fundamental_forms)
        self.H_from_components_of_fundamental_forms = self.filter_outliers_median(self.H_from_components_of_fundamental_forms)

    def calculate_pseudo_parametric_curvatures(self):
        self.K_pseudo = []
        self.H_pseudo = []

        for i, point in enumerate(self.points):

            normal = self.normals[i]
            x = point[0]
            y = point[1]

            a, b, c, d, e, f = self.C[i]

            # define symbolic equation
            h = a*(x**2) + b*(y**2) + c*(x*y) + d*(x) + e*(y) + f

            # derivatives
            hx = 2*a*x + c*y + d
            hxx = 2*a
            hy = 2*b*y + c*x + e
            hyy = 2*b
            hxy = c

            K_pseudo = (hxx*hyy - hxy**2)/(1 + hx**2 + hy**2)**2
            H_pseudo = (hxx+hyy)/2

            self.K_pseudo.append(H_pseudo)
            self.H_pseudo.append(K_pseudo)

            self.K_pseudo = self.filter_outliers_median(self.K_pseudo)
            self.H_pseudo = self.filter_outliers_median(self.H_pseudo)

    def calculate_parametric_curvatures_direct(self):
        self.K_from_components_of_fundamental_forms = []
        self.H_from_components_of_fundamental_forms = []

        for i, point in enumerate(self.points):

            normal = self.normals[i]
            x = point[0]
            y = point[1]

            a, b, c, d, e, f = self.C[i]

            # define symbolic equation
            h = a*(x**2) + b*(y**2) + c*(x*y) + d*(x) + e*(y) + f

            # derivatives
            hx = 2*a*x + c*y + d
            hxx = 2*a
            hy = 2*b*y + c*x + e
            hyy = 2*b
            hxy = c

            # first fundamental form
            E = hx**2
            F = hx*hy
            G = hy**2
            # print(f'E: {E}, F: {F}, G: {G}')

            # second fundamental form
            L = np.dot(hxx,normal[0])
            M = np.dot(hxy,normal[1])
            N = np.dot(hyy,normal[2])
            # print(f'L: {L}, M: {M}, N: {N}')

            # print(f'L: {L}, M: {M}, N: {N}')
            if (E*G-F**2) == 0 or (2*(E*G-F**2)) == 0:
                K_from_components_of_fundamental_forms = 0
                H_from_components_of_fundamental_forms = 0
            else:
                K_from_components_of_fundamental_forms = (L*N-M**2)/(E*G-F**2)
                H_from_components_of_fundamental_forms = (E*N-2*F*M+G*L)/(2*(E*G-F**2))

            self.K_from_components_of_fundamental_forms.append(K_from_components_of_fundamental_forms)
            self.H_from_components_of_fundamental_forms.append(H_from_components_of_fundamental_forms)

            self.K_from_components_of_fundamental_forms = self.filter_outliers_median(self.K_from_components_of_fundamental_forms)
            self.H_from_components_of_fundamental_forms = self.filter_outliers_median(self.H_from_components_of_fundamental_forms)

    def calculate_parametric_curvatures_symbolic(self):
        self.K_from_components_of_fundamental_forms = []
        self.H_from_components_of_fundamental_forms = []

        for i, point in enumerate(self.points):
            normal = self.normals[i]

            a, b, c, d, e, f = self.C[i]

            # define symbolic variables
            x = sympy.symbols('x', real=True)
            y = sympy.symbols('y', real=True)

            # define symbolic equation
            h = a*(x**2) + b*(y**2) + c*(x*y) + d*(x) + e*(y) + f

            # derivatives
            hx = h.diff(x)
            hxx = hx.diff(x)
            hy = h.diff(y)
            hyy = hy.diff(y)
            hxy = hx.diff(y)

            # first fundamental form
            E = np.dot(hx,hx)
            F = np.dot(hx,hy)
            G = np.dot(hy,hy)
            # print(f'E: {E}, F: {F}, G: {G}')

            # second fundamental form
            L = np.dot(hxx,normal[0])
            M = np.dot(hxy,normal[1])
            N = np.dot(hyy,normal[2])
            # print(f'L: {L}, M: {M}, N: {N}')

            E = (E.subs(x, point[0]).subs(y, point[1]))
            F = (F.subs(x, point[0]).subs(y, point[1]))
            G = (G.subs(x, point[0]).subs(y, point[1]))
            # print(f'E: {E}, F: {F}, G: {G}')

            L = (L.subs(x, point[0]).subs(y, point[1]))
            M = (M.subs(x, point[0]).subs(y, point[1]))
            N = (N.subs(x, point[0]).subs(y, point[1]))

            # print(f'L: {L}, M: {M}, N: {N}')
            if (E*G-F**2) == 0 or (2*(E*G-F**2)) == 0:
                K_from_components_of_fundamental_forms = 0
                H_from_components_of_fundamental_forms = 0
            else:
                K_from_components_of_fundamental_forms = (L*N-M**2)/(E*G-F**2)
                H_from_components_of_fundamental_forms = (E*N-2*F*M+G*L)/(2*(E*G-F**2))

            # A = np.dot(np.linalg.inv(FFF), SFF)
            # print(A)
            # k1, k2 = np.linalg.eigvals(A)

            # K_from_eigenvalues_of_invFFF_dot_SFF = k1*k2
            # H_from_eigenvalues_of_invFFF_dot_SFF = (k1+k2)/2
            
            self.K_from_components_of_fundamental_forms.append(K_from_components_of_fundamental_forms)
            self.H_from_components_of_fundamental_forms.append(H_from_components_of_fundamental_forms)

            self.K_from_components_of_fundamental_forms = self.filter_outliers_median(self.K_from_components_of_fundamental_forms)
            self.H_from_components_of_fundamental_forms = self.filter_outliers_median(self.H_from_components_of_fundamental_forms)

    def pickle_figure(self, fig, k, v):
        pickle.dump(fig, open(f'{k}_neighbors_and_{v}_voxel_size.pickle', 'wb'))

    def plot_parametric_curvatures(self):
        n = 1

        # Gaussian Curvature from fundamental form coefficients
        fig_curvature_K = plt.figure()
        ax_curvature_K = fig_curvature_K.add_subplot(111, projection='3d')
        ax_curvature_K.set_title('Gaussian Curvature from fundamental form coefficients')
        sc = ax_curvature_K.scatter(self.points[n-1:, 0], self.points[n-1:, 1], self.points[n-1:, 2], c=self.K_from_components_of_fundamental_forms, cmap='viridis', s=1)
        fig_curvature_K.colorbar(sc, ax=ax_curvature_K)
        plt.tight_layout()
        ax_curvature_K.view_init(azim=90, elev=85)
        ax_curvature_K.set_axis_off()
        ax_curvature_K.set_title(f'Gaussian Curvature from fundamental form coefficients, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')
        pickle.dump(fig_curvature_K, open(f'pcl_gaussian_curvature_from_fundamental_form coefficients_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

        # Mean Curvature from fundamental form coefficients
        fig_curvature_H = plt.figure()
        ax_curvature_H = fig_curvature_H.add_subplot(111, projection='3d')
        ax_curvature_H.set_title('Mean Curvature from fundamental form coefficients')
        sc = ax_curvature_H.scatter(self.points[n-1:, 0], self.points[n-1:, 1], self.points[n-1:, 2], c=self.H_from_components_of_fundamental_forms, cmap='viridis', s=1)
        fig_curvature_H.colorbar(sc, ax=ax_curvature_H)
        plt.tight_layout()
        ax_curvature_H.view_init(azim=90, elev=85)
        ax_curvature_H.set_axis_off()
        ax_curvature_H.set_title(f'Mean Curvature from fundamental form coefficients, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')
        pickle.dump(fig_curvature_H, open(f'pcl_mean_curvature_from_fundamental_form_coefficients_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

        # plt.show()

        # Plot histograms
        fig_hist_K_fund = plt.figure()
        plt.hist(np.array(self.K_from_components_of_fundamental_forms, dtype=float), bins=100)
        plt.title(f'Gaussian Curvature from fundamental form coefficients k={self.k_neighbors} voxel size={self.voxel_size}')
        plt.tight_layout()
        pickle.dump(fig_hist_K_fund, open(f'histogram_gaussian_curvature_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

        fig_hist_H_fund = plt.figure()
        plt.hist(np.array(self.H_from_components_of_fundamental_forms, dtype=float), bins=100)
        plt.title(f'Mean Curvature from fundamental form coefficients k={self.k_neighbors} voxel size={self.voxel_size}')
        plt.tight_layout()
        pickle.dump(fig_hist_H_fund, open(f'histogram_mean_curvature_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

        # plt.show()

    def plot_pseudo_parametric_curvatures(self):
        n = 1

        # Gaussian Curvature from fundamental form coefficients
        fig_curvature_K = plt.figure()
        ax_curvature_K = fig_curvature_K.add_subplot(111, projection='3d')
        ax_curvature_K.set_title('Pseudo Gaussian Curvature from fundamental form coefficients')
        sc = ax_curvature_K.scatter(self.points[n-1:, 0], self.points[n-1:, 1], self.points[n-1:, 2], c=self.K_pseudo, cmap='viridis', s=1)
        fig_curvature_K.colorbar(sc, ax=ax_curvature_K)
        plt.tight_layout()
        ax_curvature_K.view_init(azim=90, elev=85)
        ax_curvature_K.set_axis_off()
        ax_curvature_K.set_title(f'Pseudo Gaussian Curvature from fundamental form coefficients, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')
        pickle.dump(fig_curvature_K, open(f'pcl_pseudo_gaussian_curvature_from_fundamental_form coefficients_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

        # Mean Curvature from fundamental form coefficients
        fig_curvature_H = plt.figure()
        ax_curvature_H = fig_curvature_H.add_subplot(111, projection='3d')
        ax_curvature_H.set_title('Pseudo Mean Curvature')
        sc = ax_curvature_H.scatter(self.points[n-1:, 0], self.points[n-1:, 1], self.points[n-1:, 2], c=self.H_pseudo, cmap='viridis', s=1)
        fig_curvature_H.colorbar(sc, ax=ax_curvature_H)
        plt.tight_layout()
        ax_curvature_H.view_init(azim=90, elev=85)
        ax_curvature_H.set_axis_off()
        ax_curvature_H.set_title(f'Pseudo Mean Curvature, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')
        pickle.dump(fig_curvature_H, open(f'pcl_pseudo_mean_curvature_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

        # plt.show()

        # Plot histograms
        fig_hist_K_fund = plt.figure()
        plt.hist(np.array(self.K_pseudo, dtype=float), bins=100)
        plt.title(f'Pseudo Gaussian Curvature k={self.k_neighbors} voxel size={self.voxel_size}')
        plt.tight_layout()
        pickle.dump(fig_hist_K_fund, open(f'histogram_pseudo_gaussian_curvature_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

        fig_hist_H_fund = plt.figure()
        plt.hist(np.array(self.H_pseudo, dtype=float), bins=100)
        plt.title(f'Pseudo Mean Curvature k={self.k_neighbors} voxel size={self.voxel_size}')
        plt.tight_layout()
        pickle.dump(fig_hist_H_fund, open(f'histogram_pseudo_mean_curvature_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

        # plt.show()

    def principal_curvatures_via_principal_component_analysis(self, k_neighbors):
        num_points = len(self.points)
        principal_curvature_values_1 = np.zeros(num_points)
        principal_curvature_values_2 = np.zeros(num_points)
        principal_curvature_directions = np.zeros((num_points, 3, 2))  # Two principal directions for each point
        K_from_pca = np.zeros(num_points)
        H_from_pca = np.zeros(num_points)


        for i in range(num_points):
            point = self.points[i]

            # Find the k-nearest neighbors for the current point
            distances = np.linalg.norm(self.points - point, axis=1)
            sorted_indices = np.argsort(distances)
            neighborhood_indices = sorted_indices[1:k_neighbors + 1]  # Exclude the point itself

            # Extract the neighborhood points
            neighborhood = self.points[neighborhood_indices]

            # Calculate the covariance matrix
            covariance_matrix = np.cov(neighborhood, rowvar=False)

            # Perform eigenvalue decomposition
            eigenvalues, eigenvectors = eigh(covariance_matrix)

            # Sort eigenvalues in descending order
            sorted_indices = np.argsort(eigenvalues)[::-1]
            # eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]

            # The eigenvalues represent the principal curvatures
            principal_curvature_values_1[i] = max(eigenvalues)
            eigenvalues = np.delete(eigenvalues, np.argmax(eigenvalues))
            principal_curvature_values_2[i] = max(eigenvalues)
            principal_curvature_directions[i] = eigenvectors[:, :2]  # First two eigenvectors as principal directions

            K_from_pca[i] = principal_curvature_values_1[i]*principal_curvature_values_2[i]
            H_from_pca[i] = (principal_curvature_values_1[i]+principal_curvature_values_2[i])/2

        self.pca_principal_curvature_values_1 = principal_curvature_values_1
        self.pca_principal_curvature_values_2 = principal_curvature_values_2
        self.principal_curvature_directions = principal_curvature_directions
        self.pca_K_values = K_from_pca
        self.pca_H_values = H_from_pca

        self.pca_principal_curvature_values_1 = self.filter_outliers_median(self.pca_principal_curvature_values_1)
        self.pca_principal_curvature_values_2 = self.filter_outliers_median(self.pca_principal_curvature_values_2)
        self.pca_K_values = self.filter_outliers_median(self.pca_K_values)
        self.pca_H_values = self.filter_outliers_median(self.pca_H_values)

    def plot_principal_curvatures_from_principal_component_analysis(self):
        figg = plt.figure()
        ax3 = figg.add_subplot(111, projection='3d')
        sc3 = ax3.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c=self.pca_principal_curvature_values_1, cmap='viridis', vmin=np.mean(self.pca_principal_curvature_values_1)-2*np.std(self.pca_principal_curvature_values_1), vmax=np.mean(self.pca_principal_curvature_values_1)+2*np.std(self.pca_principal_curvature_values_1), s=10*(max(self.points[:,0])-min(self.points[:,0]))/len(self.points[:,0]))
        figg.colorbar(sc3, ax=ax3)
        ax3.set_title(f'Principal curvature 1 from PCA k={self.k_neighbors} voxel size={self.voxel_size}')
        plt.tight_layout()
        ax3.view_init(azim=90, elev=85)
        ax3.set_axis_off()
        pickle.dump(figg, open(f'principal_curvature_1_from_PCA_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

        fig7 = plt.figure()
        ax7 = fig7.add_subplot(111, projection='3d')
        sc7 = ax7.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c=self.pca_principal_curvature_values_2, cmap='viridis', vmin=np.mean(self.pca_principal_curvature_values_2)-2*np.std(self.pca_principal_curvature_values_2), vmax=np.mean(self.pca_principal_curvature_values_2)+2*np.std(self.pca_principal_curvature_values_2), s=10*(max(self.points[:,0])-min(self.points[:,0]))/len(self.points[:,0]))
        fig7.colorbar(sc7, ax=ax7)
        ax7.set_title(f'Principal curvature 2 from PCA k={self.k_neighbors} voxel size={self.voxel_size}')
        ax3.view_init(azim=90, elev=85)
        plt.tight_layout()
        ax7.set_axis_off()
        pickle.dump(fig7, open(f'principal_curvature_2_from_PCA_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

        # plt.show()

    def plot_principal_curvature_directions_from_principal_component_analysis(self):
        fig2 = plt.figure()
        ax5 = fig2.add_subplot(1, 1, 1, projection='3d')
        ax5.quiver(self.points[:, 0], self.points[:, 1], self.points[:, 2], self.principal_curvature_directions[:, 0, 0], self.principal_curvature_directions[:, 1, 0], np.zeros_like(self.points[:, 2]), length=1, normalize=True, color='g')
        ax5.set_axis_off()
        plt.title(f'Principal curvature directions (eigenvectors of covariance matrix) from PCA k={self.k_neighbors} voxel size={self.voxel_size}')
        pickle.dump(fig2, open(f'principal_curvature_vectors_from_PCA_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

        # plt.show()  

    def plot_mean_and_gaussian_curvatures_from_principal_component_analysis(self):
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(1, 1, 1, projection='3d')
        g = ax4.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c=self.pca_K_values, cmap='viridis', s=10*(max(self.points[:,0])-min(self.points[:,0]))/len(self.points[:,0]))
        plt.tight_layout()
        ax4.view_init(90, 85)
        ax4.set_axis_off()
        # fig4.colorbar(g, ax=self.pca_K_values)
        plt.title(f'Gaussian curvature from PCA k={self.k_neighbors} voxel size={self.voxel_size}')
        pickle.dump(fig4, open(f'pcl_gaussian_curvature_from_PCA_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

        

        fig0 = plt.figure()
        ax0 = fig0.add_subplot(1, 1, 1, projection='3d')
        h = ax0.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c=self.pca_H_values, cmap='viridis', s=10*(max(self.points[:,0])-min(self.points[:,0]))/len(self.points[:,0]))
        # fig0.colorbar(h, ax=self.pca_H_values)
        plt.tight_layout()
        ax0.view_init(90, 85)
        ax0.set_axis_off()
        plt.title(f'Mean curvature from PCA k={self.k_neighbors} voxel size={self.voxel_size}')
        pickle.dump(fig0, open(f'mean_curvature_from_PCA_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

        # plt.show()






