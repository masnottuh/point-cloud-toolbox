########################################################################
# Class used to process point clouds
# Robert "Sam" Hutton 
# rhutton@unr.edu or sam@samhutton.net
########################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.linalg import eigh
import sympy as sympy
from matplotlib.patches import Patch
from scipy.optimize import least_squares
import pickle
from scipy.optimize import minimize
from scipy.linalg import svd
import pymesh
import pyvista as pv
import os
import pandas as pd
import logging

class PointCloud:

    def __init__(self, file_path=None, points=None, normals=None, downsample=False, voxel_size=0, k_neighbors=20, output_path='./output/', max_points_per_voxel=1):
        self.downsample = downsample
        self.k_neighbors = k_neighbors
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.output_path = output_path
        self.random_indexes = []

        if file_path:
            self.file_path = file_path
            self.read_from_file()
        elif points is not None and normals is not None:
            self.points = points
            self.normals = normals
        else:
            raise ValueError("Either file_path or points and normals must be provided")

        self.num_points = len(self.points)
        self.num_features = len(self.points[0])
        self.l1_norm = np.linalg.norm(self.points, 1)
        self.l2_norm = np.linalg.norm(self.points, 2)
        self.infinity_norm = np.linalg.norm(self.points, np.inf)

    def read_from_file(self):
        points = np.loadtxt(self.file_path)
        self.points = points[:, 0:3]
        self.normals = points[:, 3:6]

        self.points[:, 0] -= np.max(self.points[:, 0])
        self.points[:, 1] -= np.max(self.points[:, 1])

        if self.downsample:
            self.points, self.normals = self.downsample_point_cloud_by_grid()
            self.points[:, 0] -= np.min(self.points[:, 0])
            self.points[:, 1] -= np.min(self.points[:, 1])

        self.x_domain = [np.min(self.points[:, 0]), np.max(self.points[:, 0])]
        self.y_domain = [np.min(self.points[:, 1]), np.max(self.points[:, 1])]
        self.z_domain = [np.min(self.points[:, 2]), np.max(self.points[:, 2])]


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

    def plot_surface(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c='b', s=1.5)
        ax.set_title('Point Cloud')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        pickle.dump(fig, open(f'{self.output_path}point_cloud_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

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
    def filter_outliers_median(data, threshold=100):
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
    def filter_outliers_absolute(data, max_abs=100):
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
    
    @staticmethod
    def get_best_fit_plane_and_rotate(points):
        # Check for finite values in input points
        if not np.all(np.isfinite(points)):
            raise ValueError("Non-finite values in input points")

        # Calculate the covariance matrix of the centered points
        Cov = np.cov(points, rowvar=False)

        # Perform Singular Value Decomposition
        U, S, Vt = np.linalg.svd(Cov, full_matrices=True)

        # Extract the normal vector from the last singular vector
        normal = Vt[-1]

        # Choose a reference vector as the vector from the first point to the last point in the collection
        reference_vector = points[-1] - points[0]

        # Normalize the normal and reference vectors for accurate dot product calculations
        normal_normalized = normal / np.linalg.norm(normal)
        reference_vector_normalized = reference_vector / np.linalg.norm(reference_vector)

        # Calculate the dot product between the normal and the reference vector
        dot_product = np.dot(normal_normalized, reference_vector_normalized)

        # Flip the normal if the dot product is negative
        if dot_product < 0:
            normal = -normal

        # Align the normal vector with the z-axis [0, 0, 1]
        z_axis = np.array([0, 0, 1])
        a = normal / np.linalg.norm(normal)
        b = z_axis
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)

        # Handle case where s is zero
        if s == 0:
            rotation_matrix = np.eye(3)
        else:
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

        # Rotate the points using the rotation matrix
        rotated_points = np.dot(rotation_matrix, points.T).T

        # Check for finite values in rotated points
        if not np.all(np.isfinite(rotated_points)):
            raise ValueError("Non-finite values after rotation")

        return rotated_points
    
    @staticmethod
    def plot_3d_points(points, title, ax):
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    @staticmethod
    def fit_quadratic_surface(points):
        a = points[:, 0]
        b = points[:, 1]
        c = points[:, 2]

        def quadratic_surface(params, a, b):
            A, B, C, D, E, F = params
            return A*a**2 + B*b**2 + C*a*b + D*a + E*b + F

        def objective_function(params, a, b, c):
            return quadratic_surface(params, a, b) - c

        initial_guess = np.ones(6)

        # Debugging outputs
        if not np.all(np.isfinite(a)):
            print(f"Non-finite values in a: {a}")
        if not np.all(np.isfinite(b)):
            print(f"Non-finite values in b: {b}")
        if not np.all(np.isfinite(c)):
            print(f"Non-finite values in c: {c}")

        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.all(np.isfinite(c)):
            raise ValueError("Non-finite values in points: a={}, b={}, c={}".format(a, b, c))

        result = least_squares(objective_function, initial_guess, args=(a, b, c))
        if not result.success:
            raise ValueError("Least squares optimization failed")

        return result.x
    
    @staticmethod
    def fit_implicit_quadric_surface(points):
        # Construct the design matrix for the quadratic form
        A = np.column_stack((
            points[:, 0]**2,  # x^2
            points[:, 1]**2,  # y^2
            points[:, 2]**2,  # z^2
            points[:, 0]*points[:, 1],  # xy
            points[:, 0]*points[:, 2],  # xz
            points[:, 1]*points[:, 2],  # yz
            points[:, 0],  # x
            points[:, 1],  # y
            points[:, 2],  # z
            np.ones(points.shape[0])  # Constant term
        ))

        # Define the objective function (least squares)
        def objective_function(coefficients):
            return np.sum((A @ coefficients)**2)

        # Define the constraint ||b|| - 1 = 0
        def constraint(coefficients):
            return np.linalg.norm(coefficients) - 1

        # Initial guess
        initial_guess = np.ones(A.shape[1])

        # Define the constraint as a dictionary
        cons = ({'type': 'eq', 'fun': constraint})

        # Solve the constrained minimization problem
        result = minimize(objective_function, initial_guess, constraints=cons)

        # Return the coefficients from the minimization result
        return result.x

    @staticmethod
    def calculate_explicit_quadratic_curvatures(coefficients):

        # x = point[0]
        # y = point[1]

        x = 0
        y = 0

        a, b, c, d, e, f = coefficients

        # parametric surface
        F = a*(x**2) + b*(y**2) + c*(x*y) + d*(x) + e*(y) + f

        Fx = 2*a*x + c*y + d
        Fxx = 2*a
        Fy = 2*b*y + c*x + e
        Fyy = 2*b
        Fxy = c

        # mean and gaussian curvatures 
        # https://en.wikipedia.org/wiki/Gaussian_curvature
        # https://en.wikipedia.org/wiki/Mean_curvature
        K_g = (Fxx*Fyy - Fxy**2)/((1 + Fx**2 + Fy**2)**2)
        K_h = ((1+Fx**2)*(Fyy)-2*Fx*Fy*Fxy+(1+Fy**2)*Fxx)/(2*((1+Fx**2+Fy**2)**(3/2)))
        K_h_sq = K_h**2

        # principal Curvatures
        k1 = K_h + np.sqrt(K_h**2 - K_g)
        k2 = K_h- np.sqrt(K_h**2 - K_g)

        return K_g, K_h, k1, k2, K_h_sq

    @staticmethod
    def calculate_implicit_quadric_curvatures(coefficients):

        # x = point[0]
        # y = point[1]
        # z = point[2]

        x = 0
        y = 0
        z = 0

        A, B, C, D, E, F, G, H, I, J = coefficients

        func = A*(x**2) + B*(y**2) + C*(z**2) + D*x*y + E*x*z + F*y*z + G*x + H*y + I*z + J

        # calculate the partial derivatives of the surface Ax2+By2+Cz2+Dxy+Exz+Fyz+Gx+Hy+Jz+K=0.
        fx = 2*A*x + D*y + E*z + G
        fy = 2*B*y + D*x + F*z + H
        fz = 2*C*z + E*x + F*y + I
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
        mag_g = np.sqrt(g.dot(g))
        hess_f = hessian = np.array([[fxx, fxy, fxz], [fyx, fyy, fyz], [fzx, fzy, fzz]])
        trace_hess = fxx + fyy + fzz

        hess_with_grads = np.array([[fxx, fxy, fxz, fx], [fyx, fyy, fyz, fy], [fzx, fzy, fzz, fz], [fx, fy, fz, 0]])

        # mean and gaussian curvatures 
        # https://en.wikipedia.org/wiki/Gaussian_curvature
        # https://en.wikipedia.org/wiki/Mean_curvature
        K_g = (np.linalg.det(hess_f))/(mag_g**4)
        K_h = (np.dot(np.dot(g, hess_f), g.T) - (mag_g**2)*trace_hess)/(2*(mag_g**3))
        
        # principal curvatures
        k1 = K_h + np.sqrt(K_h**2 - K_g)
        k2 = K_h- np.sqrt(K_h**2 - K_g)

        return K_g, K_h, k1, k2

    def generate_sphere_point_cloud(self, radius, num_points):
        self.points = []
        self.normals = []
        for _ in range(num_points):
            phi = np.random.uniform(0, 2 * np.pi)
            costheta = np.random.uniform(-1, 1)

            theta = np.arccos(costheta)
            r = radius  # Fixed at the sphere's radius

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            nx = x / radius  # Normal vector components (since it's a sphere, they are the same as the coordinates)
            ny = y / radius
            nz = z / radius
            
            self.points.append(np.array([x, y, z]))
            self.normals.append(np.array([nx, ny, nz]))
        self.points = np.array(self.points)
        self.normals = np.array(self.normals)

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

        pickle.dump(fig, open(f'{self.output_path}nearest_neighbors_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))
        # plt.show()

    def plot_points_colored_by_quadric_curvatures(self):

        # Gaussian Curvature from quadric calculations
        fig_curvature_K = plt.figure()
        ax_curvature_K = fig_curvature_K.add_subplot(111, projection='3d')
        sc = ax_curvature_K.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c=self.K_quadric, cmap='viridis', s=1)
        fig_curvature_K.colorbar(sc, ax=ax_curvature_K)
        plt.tight_layout()
        ax_curvature_K.view_init(azim=90, elev=85)
        ax_curvature_K.set_axis_off()
        ax_curvature_K.set_title(f'Gaussian Curvature from quadric surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')
        pickle.dump(fig_curvature_K, open(f'{self.output_path}Gaussian Curvature from quadric surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}.pickle', 'wb'))

        # Mean Curvature from quadric surface
        fig_curvature_H = plt.figure()
        ax_curvature_H = fig_curvature_H.add_subplot(111, projection='3d')
        sc = ax_curvature_H.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c=self.H_quadric, cmap='viridis', s=1)
        fig_curvature_H.colorbar(sc, ax=ax_curvature_H)
        plt.tight_layout()
        ax_curvature_H.view_init(azim=90, elev=85)
        ax_curvature_H.set_axis_off()
        ax_curvature_H.set_title(f'Mean Curvature from quadric surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')
        pickle.dump(fig_curvature_H, open(f'{self.output_path}Mean Curvature from quadric surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}.pickle', 'wb'))

        # plt.show()

        # # Plot histograms
        # fig_hist_K_fund = plt.figure()
        # plt.hist(np.array(self.K_quadric, dtype=float), bins=100)
        # plt.title(f'Hist Gaussian Curvature from quadric surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')
        # plt.tight_layout()
        # pickle.dump(fig_hist_K_fund, open(f'{self.output_path}Hist Gaussian Curvature from quadric surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}.pickle', 'wb'))

        # fig_hist_H_fund = plt.figure()
        # plt.hist(np.array(self.H_quadric, dtype=float), bins=100)
        # plt.title(f'Hist Mean Curvature from quadric surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')
        # plt.tight_layout()
        # pickle.dump(fig_hist_H_fund, open(f'{self.output_path}Hist Mean Curvature from quadric surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}.pickle', 'wb'))

        # plt.show()

    def plot_points_colored_by_quadratic_curvatures(self):
        # self.filter_outlier_curvatures_per_neighborhood()

        # Gaussian Curvature from quadric calculations
        fig_curvature_K = plt.figure()
        ax_curvature_K = fig_curvature_K.add_subplot(111, projection='3d')
        sc = ax_curvature_K.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c=self.K_quadratic, cmap='viridis', s=1)
        fig_curvature_K.colorbar(sc, ax=ax_curvature_K)
        plt.tight_layout()
        ax_curvature_K.view_init(azim=90, elev=85)
        ax_curvature_K.set_axis_off()
        ax_curvature_K.set_title(f'Gaussian Curvature from quadratic surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')
        pickle.dump(fig_curvature_K, open(f'{self.output_path}Gaussian Curvature from quadratic surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}.pickle', 'wb'))

        # Mean Curvature from quadric surface
        fig_curvature_H = plt.figure()
        ax_curvature_H = fig_curvature_H.add_subplot(111, projection='3d')
        sc = ax_curvature_H.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c=self.H_quadratic, cmap='viridis', s=1)
        fig_curvature_H.colorbar(sc, ax=ax_curvature_H)
        plt.tight_layout()
        ax_curvature_H.view_init(azim=90, elev=85)
        ax_curvature_H.set_axis_off()
        ax_curvature_H.set_title(f'Mean Curvature from quadratic surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')
        pickle.dump(fig_curvature_H, open(f'{self.output_path}Mean Curvature from quadratic surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}.pickle', 'wb'))


        # Mean Curvature squared from quadric surface
        fig_curvature_H2 = plt.figure()
        ax_curvature_H2 = fig_curvature_H2.add_subplot(111, projection='3d')
        sc2 = ax_curvature_H2.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c=self.K_H_sq_quadratic, cmap='viridis', s=1)
        fig_curvature_H2.colorbar(sc2, ax=ax_curvature_H2)
        plt.tight_layout()
        ax_curvature_H2.view_init(azim=90, elev=85)
        ax_curvature_H2.set_axis_off()
        ax_curvature_H2.set_title(f'Mean Curvature squared from quadratic surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')
        pickle.dump(fig_curvature_H2, open(f'{self.output_path}Mean Curvature Squared from quadratic surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}.pickle', 'wb'))

        # plt.show()

        # # Plot histograms
        # fig_hist_K_fund = plt.figure()
        # plt.hist(np.array(self.K_quadratic, dtype=float), bins=100)
        # plt.title(f'Hist Gaussian Curvature from quadratic surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')
        # plt.tight_layout()
        # pickle.dump(fig_hist_K_fund, open(f'{self.output_path}Hist Gaussian Curvature from quadratic surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}.pickle', 'wb'))

        # fig_hist_H_fund = plt.figure()
        # plt.hist(np.array(self.H_quadratic, dtype=float), bins=100)
        # plt.title(f'Hist Mean Curvature from quadratic surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}')
        # plt.tight_layout()
        # pickle.dump(fig_hist_H_fund, open(f'{self.output_path}Hist Mean Curvature from quadratic surface, K = {self.k_neighbors}, Voxel Size = {self.voxel_size}.pickle', 'wb'))
        # plt.close()
        # plot(self.points, c=np.array(self.K_quadratic), shading={"point_size": 10.0})
        # plot(self.points, c=np.array(self.H_quadratic), shading={"point_size": 10.0})
        # mp.subplot(self.points, c=np.random.rand(*v.shape), s=[1, 2, 1], data=d, shading={"point_size": 0.03})

        # plt.show()

    def fit_implicit_quadric_surfaces_all_points(self):

        self.quadric_coefficients = [[] for i in range(len(self.points))]

        for i, point in enumerate(self.points):
            
            k = self.k_neighbors
            
            dists, neighbor_indicies = self.kdtree.query(point, k)
            neighbors = self.points[neighbor_indicies]

            # center the points to the origin
            neighbors = neighbors - point

            points = neighbors
            coefficients = self.fit_implicit_quadric_surface(points)
            self.quadric_coefficients[i] = coefficients

    def fit_explicit_quadratic_surfaces_to_neighborhoods(self):
    
        self.quadratic_coefficients = [[] for i in range(len(self.points))]
        for i, point in enumerate(self.points):

            points = self.points[self.neighbor_indices[i]]
            centered_points = points - point

            # Find the best-fit plane using SVD
            points = self.get_best_fit_plane_and_rotate(centered_points)

            # Fit a quadratic surface to the rotated points
            self.quadratic_coefficients[i] = self.fit_quadratic_surface(points)

    @staticmethod
    def calculate_energies(voronoi_areas, gaussian_curvature, mean_curvature):
        # Calculate bending and stretching energies
        bending_energy = sum((h ** 2) * area for h, area in zip(mean_curvature, voronoi_areas))
        stretching_energy = sum(k * area for k, area in zip(gaussian_curvature, voronoi_areas))

        return bending_energy, stretching_energy

    def calculate_curvatures_of_explicit_quadratic_surfaces_for_all_points(self):

        self.K_quadratic = []
        self.H_quadratic = []
        self.K_H_sq_quadratic = []

        for i, point in enumerate(self.points):

            normal = self.normals[i]
            coefs = self.quadratic_coefficients[i]

            K_g, K_h, k1, k2, K_h_sq = self.calculate_explicit_quadratic_curvatures(coefs)

            self.K_quadratic.append(K_g)
            self.H_quadratic.append(K_h)
            self.K_H_sq_quadratic.append(K_h_sq)
        
        return self.K_quadratic, self.H_quadratic

    def calculate_curvatures_of_implicit_quadric_surfaces_for_all_points(self):

        self.K_quadric = []
        self.H_quadric = []

        for i, point in enumerate(self.points):

            normal = self.normals[i]
            coefs = self.quadric_coefficients[i]

            K_g, K_h, k1, k2 = self.calculate_implicit_quadric_curvatures(coefs)

            self.K_quadric.append(K_g)
            self.H_quadric.append(K_h)

    def compute_normals(self):
        """Compute normals for the point cloud and store them in self.normals."""
        cloud = pv.PolyData(self.points)
        if cloud.n_points > 0 and cloud.n_cells == 0:
            cloud = cloud.delaunay_2d()
        cloud.compute_normals(point_normals=True, cell_normals=False, inplace=True)
        self.normals = cloud.point_data['normals']

    def export_ply_with_curvature_and_normals(self, filename):
        """Export the point cloud to a PLY file including normals, curvatures, and face data."""
        if self.normals is None:
            self.compute_normals()
        
        # Create a PyVista point cloud object
        cloud = pv.PolyData(self.points)
        cloud.point_data['normals'] = self.normals
        cloud.point_data['gaussian_curvature'] = self.K_quadratic
        cloud.point_data['mean_curvature'] = self.H_quadratic

        # Ensure faces are attached if they exist
        if hasattr(self, 'faces') and self.faces is not None and len(self.faces) > 0:
            # Flatten and reshape faces to match the PyVista format
            flat_faces = np.hstack([[len(face)] + list(face) for face in self.faces])
            cloud.faces = flat_faces
            print(f"Number of Faces being added: {len(self.faces)}")
        else:
            print("No valid faces found, not adding face data to the PolyData object.")

        # Debugging: Check what's in point_data before saving
        print("Point Data Keys:", cloud.point_data.keys())
        print("Number of Points:", len(self.points))
        print("Number of Faces:", len(cloud.faces) if hasattr(cloud, 'faces') else 'No Faces')

        # Save the point cloud as a PLY file
        cloud.save(filename, binary=False)  # Use binary=False for ASCII format
        print(f"Point cloud saved successfully in PLY format as {filename}")



   

    def explicit_quadratic_neighbor_study(self):
        logging.info("Inside explicit_quadratic_neighbor_study()")
        points = self.points

        self.random_indexes = np.random.randint(0, len(self.points), len(self.points)//6)
        random_indexes = self.random_indexes
        random_points = self.points[random_indexes]

        test_results = {}
        explicit_converged_neighbors = []

        for i, point in enumerate(random_points):
            test_results[point[0]] = {}
            test_results[point[0]]['gaussian'] = []
            test_results[point[0]]['mean'] = []
            test_results[point[0]]['principal_1'] = []
            test_results[point[0]]['principal_2'] = []
            test_results[point[0]]['neighbors'] = []

            error_points = 0
            for num_neighbors in range(3, 99):
                neighbor_inds = self.kdtree.query(point, num_neighbors+1)[1]
                points = self.points[neighbor_inds]
                centered_points = points - point

                points = self.get_best_fit_plane_and_rotate(centered_points)
                try:
                    coefs = self.fit_quadratic_surface(points)
                except:
                    logging.info("Fitting EQNS Failure")
                    error_points+=1
                    coefs = 0, 0, 0, 0, 0, 0
                
                normal = self.normals[i]
                K_g, K_h, k1, k2, _ = self.calculate_explicit_quadratic_curvatures(coefs)

                test_results[point[0]]['neighbors'].append(num_neighbors)
                test_results[point[0]]['gaussian'].append(K_g)
                test_results[point[0]]['mean'].append(K_h)
                test_results[point[0]]['principal_1'].append(k1)
                test_results[point[0]]['principal_2'].append(k2)

                # width_of_window = 24 #max of num_neighbors//4 
    
                if len(test_results[point[0]]['gaussian']) > 10:  #Have to skip first iteration
                    maxk = max(test_results[point[0]]['gaussian'][-6:]) #Using gaussian to ensure convergence in principal_1 and p2 directions
                    mink = min(test_results[point[0]]['gaussian'][-6:])
                    difference = abs(round(maxk - mink))
                    if difference < 1e-5:
                        explicit_converged_neighbors.append(num_neighbors)
                        break
                    else:
                        pass    

        converged_neighbors_int = (sum(explicit_converged_neighbors)//len(explicit_converged_neighbors)) + 1 #Plus one cause int div rounds down 
        return converged_neighbors_int

            #To Sam: THESE PLOTTING FEATURES WERE HERE BEFORE BUT I DONT KNOW HOW TO USE IT -Gavin

            #plot test results
            # figy = plt.figure()
            # axx = figy.add_subplot(1, 1, 1)
            # axx.set_title(f'Neighbor Test, Voxel Size = {self.voxel_size}, pcl has total of {len(self.points[:,0])} points')
            # axx.set_xlabel('num neighbors')
            # axx.set_ylabel('Gaussian Curvature (quadratic fit)')
            # axx.scatter(test_results[point[0]]['neighbors'],test_results[point[0]]['gaussian'], c='b', s=2)
            # plt.text(x=1,y=1,s=f'{file_path}')
            # figy.show()
            # pickle.dump(figy, open(f'{self.output_path}Gaussian Curvature {i} study, Voxel Size = {self.voxel_size}.pickle', 'wb'))

            # figz = plt.figure()
            # axxa = figz.add_subplot(1, 1, 1)
            # axxa.set_title(f'Neighbor Test, Voxel Size = {self.voxel_size}, pcl has total of {len(self.points[:,0])} points')
            # axxa.set_xlabel('num neighbors')
            # axxa.set_ylabel('Mean Curvature (quadratic fit)')
            # axxa.scatter(test_results[point[0]]['neighbors'],test_results[point[0]]['mean'], c='b', s=2)
            # plt.text(x=1,y=1,s=f'{file_path}')
            # figz.show()
            # pickle.dump(figz, open(f'{self.output_path}Mean Curvature study, Voxel Size = {self.voxel_size} {i}.pickle', 'wb'))


    def implicit_quadric_neighbor_study(self):
        points = self.points
        # normals = self.normals
        random_indexes = self.random_indexes
        # random_points = self.random_points

        test_results = {}

        for i, point in enumerate(random_points):
            test_results[point[0]] = {}
            test_results[point[0]]['gaussian'] = []
            test_results[point[0]]['mean'] = []
            test_results[point[0]]['principal_1'] = []
            test_results[point[0]]['principal_2'] = []
            test_results[point[0]]['neighbors'] = []

            for num_neighbors in range(2, 100):
                neighbor_inds = self.kdtree.query(point, num_neighbors+1)[1]
                points = self.points[neighbor_inds]
                centered_points = points - point

                points = self.get_best_fit_plane_and_rotate(centered_points)
                try:
                    coefs = self.fit_implicit_quadric_surface(points)
                except:
                    error_points+=1
                    ceofs = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                
                K_g, K_h, k1, k2 = self.calculate_implicit_quadric_curvatures(coefs)

                test_results[point[0]]['neighbors'].append(num_neighbors)
                test_results[point[0]]['gaussian'].append(K_g)
                test_results[point[0]]['mean'].append(K_h)
                test_results[point[0]]['principal_1'].append(k1)
                test_results[point[0]]['principal_2'].append(k2)

        for i, point in enumerate(self.random_points):
            #plot test results
            figy = plt.figure()
            axx = figy.add_subplot(1, 1, 1)
            axx.set_title(f'Neighbor Test, Voxel Size = {self.voxel_size}, pcl has total of {len(self.points[:,0])} points')
            axx.set_xlabel('num neighbors')
            axx.set_ylabel('Gaussian Curvature (quadic fit)')
            axx.scatter(test_results[point[0]]['neighbors'],test_results[point[0]]['gaussian'], c='b', s=2)
            
            pickle.dump(figy, open(f'{self.output_path}Gaussian Curvature {i} study, Voxel Size = {self.voxel_size}.pickle', 'wb'))

            figz = plt.figure()
            axxa = figz.add_subplot(1, 1, 1)
            axxa.set_title(f'Neighbor Test, Voxel Size = {self.voxel_size}, pcl has total of {len(self.points[:,0])} points')
            axxa.set_xlabel('num neighbors')
            axxa.set_ylabel('Mean Curvature (quadric fit)')
            axxa.scatter(test_results[point[0]]['neighbors'],test_results[point[0]]['mean'], c='b', s=2)
            
            pickle.dump(figz, open(f'{self.output_path}Mean Curvature study, Voxel Size = {self.voxel_size} {i}.pickle', 'wb'))

    def calculate_energies(self, mesh_path):
        # Load mesh
        mesh = pymesh.load_mesh(mesh_path)
        
        # Compute curvatures
        pymesh.curvature(mesh)
        gaussian_curvature = mesh.get_attribute("vertex_gaussian_curvature")
        mean_curvature = mesh.get_attribute("vertex_mean_curvature")
        
        # Compute dual area (similar to Voronoi area)
        dual_area = mesh.get_attribute("vertex_dual_area")
        
        # Calculate energies
        bending_energy = sum((H**2 * area for H, area in zip(mean_curvature, dual_area)))
        stretching_energy = sum((K * area for K, area in zip(gaussian_curvature, dual_area)))

        return bending_energy, stretching_energy

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

        # self.pca_principal_curvature_values_1 = self.filter_outliers_median(self.pca_principal_curvature_values_1)
        # self.pca_principal_curvature_values_2 = self.filter_outliers_median(self.pca_principal_curvature_values_2)
        # self.pca_K_values = self.filter_outliers_median(self.pca_K_values)
        # self.pca_H_values = self.filter_outliers_median(self.pca_H_values)

    def plot_principal_curvatures_from_principal_component_analysis(self):
        figg = plt.figure()
        ax3 = figg.add_subplot(111, projection='3d')
        sc3 = ax3.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c=self.pca_principal_curvature_values_1, cmap='viridis', vmin=np.mean(self.pca_principal_curvature_values_1)-2*np.std(self.pca_principal_curvature_values_1), vmax=np.mean(self.pca_principal_curvature_values_1)+2*np.std(self.pca_principal_curvature_values_1), s=10*(max(self.points[:,0])-min(self.points[:,0]))/len(self.points[:,0]))
        figg.colorbar(sc3, ax=ax3)
        ax3.set_title(f'Principal curvature 1 from PCA k={self.k_neighbors} voxel size={self.voxel_size}')
        plt.tight_layout()
        ax3.view_init(azim=90, elev=85)
        ax3.set_axis_off()
        pickle.dump(figg, open(f'{self.output_path}principal_curvature_1_from_PCA_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

        fig7 = plt.figure()
        ax7 = fig7.add_subplot(111, projection='3d')
        sc7 = ax7.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c=self.pca_principal_curvature_values_2, cmap='viridis', vmin=np.mean(self.pca_principal_curvature_values_2)-2*np.std(self.pca_principal_curvature_values_2), vmax=np.mean(self.pca_principal_curvature_values_2)+2*np.std(self.pca_principal_curvature_values_2), s=10*(max(self.points[:,0])-min(self.points[:,0]))/len(self.points[:,0]))
        fig7.colorbar(sc7, ax=ax7)
        ax7.set_title(f'Principal curvature 2 from PCA k={self.k_neighbors} voxel size={self.voxel_size}')
        ax3.view_init(azim=90, elev=85)
        plt.tight_layout()
        ax7.set_axis_off()
        pickle.dump(fig7, open(f'{self.output_path}principal_curvature_2_from_PCA_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

        # plt.show()

    def plot_principal_curvature_directions_from_principal_component_analysis(self):
        fig2 = plt.figure()
        ax5 = fig2.add_subplot(1, 1, 1, projection='3d')
        ax5.quiver(self.points[:, 0], self.points[:, 1], self.points[:, 2], self.principal_curvature_directions[:, 0, 0], self.principal_curvature_directions[:, 1, 0], np.zeros_like(self.points[:, 2]), length=1, normalize=True, color='g')
        ax5.set_axis_off()
        plt.title(f'Principal curvature directions (eigenvectors of covariance matrix) from PCA k={self.k_neighbors} voxel size={self.voxel_size}')
        pickle.dump(fig2, open(f'{self.output_path}principal_curvature_vectors_from_PCA_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

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
        pickle.dump(fig4, open(f'{self.output_path}pcl_gaussian_curvature_from_PCA_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))

        

        fig0 = plt.figure()
        ax0 = fig0.add_subplot(1, 1, 1, projection='3d')
        h = ax0.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c=self.pca_H_values, cmap='viridis', s=10*(max(self.points[:,0])-min(self.points[:,0]))/len(self.points[:,0]))
        # fig0.colorbar(h, ax=self.pca_H_values)
        plt.tight_layout()
        ax0.view_init(90, 85)
        ax0.set_axis_off()
        plt.title(f'Mean curvature from PCA k={self.k_neighbors} voxel size={self.voxel_size}')
        pickle.dump(fig0, open(f'{self.output_path}mean_curvature_from_PCA_k_{self.k_neighbors}_voxel_size_{self.voxel_size}.pickle', 'wb'))


        # plt.show()
