import numpy as np
import matplotlib.pyplot as plt
from pointCloudToolbox import PointCloud
from scipy.optimize import least_squares

def generate_sphere_point_cloud(radius, num_points):
    points = []
    for _ in range(num_points):
        phi = np.random.uniform(0, 2 * np.pi)
        costheta = np.random.uniform(-1, 1)
        theta = np.arccos(costheta)
        r = radius
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        points.append([x, y, z])
    return np.array(points)

def generate_torus_point_cloud(R, r, num_points):
    points = []
    for _ in range(num_points):
        u = np.random.uniform(0, 2 * np.pi)
        v = np.random.uniform(0, 2 * np.pi)
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        points.append([x, y, z])
    return np.array(points)

def generate_cylinder_point_cloud(radius, height, num_points):
    points = []
    for _ in range(num_points):
        theta = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(-height / 2, height / 2)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    return np.array(points)

def generate_plane_point_cloud(size, num_points):
    points = np.random.uniform(-size / 2, size / 2, (num_points, 3))
    points[:, 2] = 0  # z-coordinate is 0 for a plane
    return points

def generate_saddle_point_cloud(size, num_points):
    points = np.random.uniform(-size / 2, size / 2, (num_points, 2))
    x, y = points[:, 0], points[:, 1]
    z = x**2 - y**2
    points = np.column_stack((x, y, z))
    return points

def save_ply(points, gaussian_curvature, mean_curvature, file_path):
    with open(file_path, 'w') as file:
        # Write header
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property float gaussian_curvature\n")
        file.write("property float mean_curvature\n")
        file.write("end_header\n")
        # Write data
        for i in range(len(points)):
            file.write(f"{points[i][0]} {points[i][1]} {points[i][2]} {gaussian_curvature[i]} {mean_curvature[i]}\n")

def compute_and_save_curvatures():
    # Parameters
    radius = 10
    num_points = 100000
    k_neighbors = 200

    # Sphere
    sphere_points = generate_sphere_point_cloud(radius, num_points)
    sphere_normals = sphere_points / radius
    sphere_pcl = PointCloud(points=sphere_points, normals=sphere_normals, k_neighbors=k_neighbors, output_path='./output/sphere/')
    sphere_pcl.plant_kdtree(k_neighbors)
    sphere_pcl.fit_explicit_quadratic_surfaces_to_neighborhoods()
    gaussian_curvature, mean_curvature = sphere_pcl.calculate_curvatures_of_explicit_quadratic_surfaces_for_all_points()
    save_ply(sphere_points, gaussian_curvature, mean_curvature, 'output_with_curvatures_sphere.ply')

    # Torus
    R = 15
    r = 5
    torus_points = generate_torus_point_cloud(R, r, num_points)
    torus_normals = torus_points / np.linalg.norm(torus_points, axis=1)[:, None]
    torus_pcl = PointCloud(points=torus_points, normals=torus_normals, k_neighbors=k_neighbors, output_path='./output/torus/')
    torus_pcl.plant_kdtree(k_neighbors)
    torus_pcl.fit_explicit_quadratic_surfaces_to_neighborhoods()
    gaussian_curvature, mean_curvature = torus_pcl.calculate_curvatures_of_explicit_quadratic_surfaces_for_all_points()
    save_ply(torus_points, gaussian_curvature, mean_curvature, 'output_with_curvatures_torus.ply')

    # Cylinder
    cylinder_points = generate_cylinder_point_cloud(radius, 20, num_points)
    cylinder_normals = np.zeros_like(cylinder_points)
    cylinder_normals[:, 0] = cylinder_points[:, 0] / radius
    cylinder_normals[:, 1] = cylinder_points[:, 1] / radius
    cylinder_pcl = PointCloud(points=cylinder_points, normals=cylinder_normals, k_neighbors=k_neighbors, output_path='./output/cylinder/')
    cylinder_pcl.plant_kdtree(k_neighbors)
    cylinder_pcl.fit_explicit_quadratic_surfaces_to_neighborhoods()
    gaussian_curvature, mean_curvature = cylinder_pcl.calculate_curvatures_of_explicit_quadratic_surfaces_for_all_points()
    save_ply(cylinder_points, gaussian_curvature, mean_curvature, 'output_with_curvatures_cylinder.ply')

    # Plane
    plane_points = generate_plane_point_cloud(20, num_points)
    plane_normals = np.zeros_like(plane_points)
    plane_normals[:, 2] = 1
    plane_pcl = PointCloud(points=plane_points, normals=plane_normals, k_neighbors=k_neighbors, output_path='./output/plane/')
    plane_pcl.plant_kdtree(k_neighbors)
    plane_pcl.fit_explicit_quadratic_surfaces_to_neighborhoods()
    gaussian_curvature, mean_curvature = plane_pcl.calculate_curvatures_of_explicit_quadratic_surfaces_for_all_points()
    save_ply(plane_points, gaussian_curvature, mean_curvature, 'output_with_curvatures_plane.ply')

    # Saddle
    saddle_points = generate_saddle_point_cloud(10, num_points)
    saddle_normals = np.zeros_like(saddle_points)
    saddle_normals[:, 0] = -2 * saddle_points[:, 0]
    saddle_normals[:, 1] = 2 * saddle_points[:, 1]
    saddle_normals[:, 2] = 1
    saddle_normals /= np.linalg.norm(saddle_normals, axis=1)[:, None]
    saddle_pcl = PointCloud(points=saddle_points, normals=saddle_normals, k_neighbors=k_neighbors, output_path='./output/saddle/')
    saddle_pcl.plant_kdtree(k_neighbors)
    saddle_pcl.fit_explicit_quadratic_surfaces_to_neighborhoods()
    gaussian_curvature, mean_curvature = saddle_pcl.calculate_curvatures_of_explicit_quadratic_surfaces_for_all_points()
    save_ply(saddle_points, gaussian_curvature, mean_curvature, 'output_with_curvatures_saddle.ply')

if __name__ == '__main__':
    compute_and_save_curvatures()