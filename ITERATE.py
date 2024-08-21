import pyvista as pv
import numpy as np
import tempfile
import open3d as o3d
import logging
from scipy.spatial import KDTree

import random

logging.basicConfig(level=logging.INFO)

def generate_torus(num_points, tube_radius, cross_section_radius):
    logging.info("Inside generate_torus()")
    theta = np.linspace(0, 2 * np.pi, int(np.sqrt(num_points)))
    phi = np.linspace(0, 2 * np.pi, int(np.sqrt(num_points)))

    theta, phi = np.meshgrid(theta, phi)
    theta, phi = theta.flatten(), phi.flatten()
    x = (tube_radius + cross_section_radius * np.cos(phi)) * np.cos(theta)
    y = (tube_radius + cross_section_radius * np.cos(phi)) * np.sin(theta)
    z = cross_section_radius * np.sin(phi)
    points = np.vstack((x, y, z)).T
    logging.info("Exiting generate_torus()")
    return points

def average_distance_using_kd_tree(pcd):

    logging.info("Calculating average distance between points")

    # Convert Open3D PointCloud to a numpy array
    points = np.asarray(pcd.points)
    num_points = points.shape[0]
    
    if num_points < 2:
        raise ValueError("Point cloud must contain at least two points.")
    # Create a KDTree for efficient nearest neighbor search
    tree = KDTree(points)
    total_distance = 0
    total_pairs = 0
    K = 2

    for i in range(500):
        # Query the K nearest neighbors, including the point itself
        distances, _ = tree.query(random.choice(points), k=K)
        # Exclude the distance to itself (which is always 0)
        distance = distances[1]
        print(distance)
        
        total_distance += distance
        total_pairs += 1

    average_distance = total_distance / total_pairs if total_pairs > 0 else 0
    return average_distance, total_pairs

def save_points_to_ply(points, filename):
    logging.info("Inside save_points_to_ply()")
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        np.savetxt(f, points, fmt='%.6f %.6f %.6f')
    logging.info("Exiting save_points_to_ply()")

def parse_ply(file_path):
    logging.info("Inside parse_ply()")
    try:
        with open(file_path, 'r') as file:
            # Read header
            while True:
                line = file.readline().strip()
                if line == "end_header":
                    logging.info(f"Removed header from PLY")
                    break
            # Read body data
            points = []
            while True:
                line = file.readline()
                if not line:
                    break
                parts = line.split()
                x, y, z = map(float, parts[:3])
                points.append([x, y, z])
        return np.array(points)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error parsing PLY file: {e}")


def radii_list_fun(average_distance):

    radii_list = np.linspace(average_distance,10*average_distance,3)

    return radii_list



def create_mesh(file_path, k_neighbors, radii_list):
    logging.info("Inside create_mesh()")
    points = parse_ply(file_path)
    if points is None:
        return None

    o3d.visualization.draw_geometries([pcd])
    # Estimate normals
    logging.info("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radii_list[0], len(pcd.points)//100))
    pcd.normalize_normals()

    # Remove statistical outliers
    # logging.info("Removing outliers...")
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # pcd = pcd.select_by_index(ind)

    # Perform Ball-Pivoting Algorithm (BPA) for mesh reconstruction
    logging.info("Using Ball-Pivoting Algorithm (BPA) for reconstruction...")
    # mid = len(radii_list)//2
    # for i in range((len(radii_list)//2)-1):
    # radii = [radii_list[mid-i],radii_list[mid],radii_list[mid+i]]

    radii = radii_list
    cloud = pv.PolyData(np.array(pcd.points))

    mesh = cloud.reconstruct_surface()
    # Extract the vertices and faces from the PyVista mesh
    vertices = np.array(mesh.points)
    faces = np.array(mesh.faces).reshape((-1, 4))[:, 1:]

    # Create an Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    

    # mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh).fill_holes(hole_size=100*radii_list[1]).to_legacy()
    logging.info("BPA reconstruction completed. Number of vertices: %d", len(mesh.vertices))
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, mesh_show_wireframe=True)
    # Convert Open3D mesh to PyVista mesh
    BPA_pv_mesh = pv.PolyData(np.asarray(mesh.vertices), np.hstack([[3] + face.tolist() for face in np.asarray(mesh.triangles)]))
    BPA_pv_mesh.save(f"BPA.ply")
    logging.info(f"Saved BPA.ply")
    # BPA_pv_mesh.plot(point_size=1, text='Mesh Made By BPA')
    plotter = pv.Plotter(off_screen=True)
    actor = plotter.add_mesh(BPA_pv_mesh)
    
    plotter.screenshot(f'BPA')
    
    # Save the vertices to a temporary text file that PointCloud can read
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
        np.savetxt(temp_file.name, BPA_pv_mesh.points)


    # Estimate normals
    logging.info("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.normalize_normals()

    # Remove statistical outliers
    logging.info("Removing outliers...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)

    # Perform Ball-Pivoting Algorithm (BPA) for mesh reconstruction
    logging.info("Using Ball-Pivoting Algorithm (BPA) for reconstruction...")
    mid = len(radii_list)//2
    for i in range((len(radii_list)//2)-1):
        radii = [radii_list[mid-i],radii_list[mid],radii_list[mid+i]]
        BPAmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        logging.info("BPA reconstruction completed. Number of vertices: %d", len(BPAmesh.vertices))
    # Convert Open3D mesh to PyVista mesh
        BPA_pv_mesh = pv.PolyData(np.asarray(BPAmesh.vertices), np.hstack([[3] + face.tolist() for face in np.asarray(BPAmesh.triangles)]))
        BPA_pv_mesh.save(f"BPA.ply")
        logging.info(f"Saved BPA.ply")
        # BPA_pv_mesh.plot(point_size=1, text='Mesh Made By BPA')   
        plotter = pv.Plotter(off_screen=True)
        actor = plotter.add_mesh(BPA_pv_mesh)
        plotter.screenshot(f'{i}th BPA')
    
    # Save the vertices to a temporary text file that PointCloud can read
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
        np.savetxt(temp_file.name, pv_mesh.points)



file_path = 'output_with_curvatures_torus.ply'
# torus_points = generate_torus(num_points=10000, tube_radius=10, cross_section_radius=3)
points = parse_ply(file_path)


torus_points = generate_torus(num_points=300, tube_radius=10, cross_section_radius=3)
points = parse_ply("torus.ply")
if points is None:
        print("No points")


# save_points_to_ply(points, file_name)


create_mesh(file_path, 825, radii_list,points)

save_points_to_ply(torus_points, 'torus.ply')


file_path = 'C:/Users/Lab PC/Desktop/Gavin_Fisher/PointCloudToolbox/torus.ply'
create_mesh(file_path, 825, radii_list)

logging.info("Program Done")