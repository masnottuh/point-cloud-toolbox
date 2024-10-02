import pyvista as pv
import numpy as np
import tempfile
import open3d as o3d
import logging
from scipy.spatial import KDTree
import random
from pointCloudToolbox import *
import copy
import itertools


def create_mesh_with_curvature(file_path):
    logging.info("Inside create_mesh_with_curvature()")

    # Parse the PLY file
    points = parse_ply(file_path)
    if points is None:
        return None

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Visualize the input point cloud
    o3d.visualization.draw_geometries([pcd], window_name="Input Point Cloud", mesh_show_back_face=True)

    # Calculate average distance using KDTree and get radii list
    mets = average_distance_using_kd_tree(pcd)
    radii_list = mets['radii_list']
    
    # Estimate normals for the point cloud
    logging.info("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radii_list[0], len(pcd.points) // 100))
    logging.info("Normalizing normals...")
    pcd.normalize_normals()

    # Rotate point cloud on multiple axes to create multiple meshes and merge them
    logging.info("Creating multiple BPA meshes with rotations on multiple axes...")
    meshes = []
    # Generate 5 increments between 0 and pi
    incrementsX = np.linspace(0, 0.25*np.pi, 2)
    incrementsY = np.linspace(0, 0.25*np.pi, 2)
    incrementsZ = np.linspace(0, np.pi, 20)

    # Create combinations of these increments for rotation around X, Y, and Z axes
    rotation_axes = list(itertools.product(incrementsX, incrementsY, incrementsZ))

    for angles in rotation_axes:
        logging.info(f"Rotating point cloud by angles: {angles}")

        # Use deepcopy to clone the point cloud instead of the clone() method
        rotated_pcd = copy.deepcopy(pcd)
        
        # Rotate the cloned point cloud on specified axes
        rotation_matrix = rotated_pcd.get_rotation_matrix_from_xyz(angles)
        rotated_pcd.rotate(rotation_matrix, center=(0, 0, 0))  # Rotate around origin

        # Create mesh using Ball Pivoting Algorithm (BPA) on the rotated cloned point cloud
        logging.info("Creating mesh using BPA on rotated point cloud...")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(rotated_pcd, o3d.utility.DoubleVector(radii_list))

        # Rotate the mesh back to the original orientation
        logging.info("Rotating mesh back to the original orientation...")
        inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
        mesh.rotate(inverse_rotation_matrix, center=(0, 0, 0))  # Rotate back around the same origin

        meshes.append(mesh)

    # Combine all the meshes
    logging.info("Combining BPA meshes...")
    final_mesh = meshes[0]
    for mesh in meshes[1:]:
        final_mesh += mesh

    # Post-process the merged mesh: Remove degenerate and non-manifold elements
    logging.info("Post-processing mesh: removing degenerate triangles and non-manifold edges...")
    final_mesh.remove_degenerate_triangles()
    final_mesh.remove_duplicated_vertices()
    final_mesh.remove_non_manifold_edges()

    # Vertex clustering step to further simplify the mesh
    logging.info("Applying vertex clustering for mesh simplification...")
    voxel_size = radii_list[0]  # Adjust as needed for your mesh scale
    final_mesh = final_mesh.simplify_vertex_clustering(voxel_size=voxel_size)


    # Visualize the final mesh
    logging.info("Visualizing the final mesh...")
    o3d.visualization.draw_geometries([final_mesh], mesh_show_back_face=True)

    # Extract vertices and faces from the filled Open3D mesh
    logging.info("Extracting vertices and faces from Open3D mesh...")
    vertices = np.asarray(final_mesh.vertices)
    faces = np.asarray(final_mesh.triangles)

    # Convert Open3D mesh to PyVista format
    logging.info("Converting Open3D mesh to PyVista mesh...")
    pv_mesh = pv.PolyData(vertices, np.hstack([[3] + face.tolist() for face in faces]))

    # fill remaining holes
    pv_mesh = pv_mesh.fill_holes(3*radii_list[-1])

    # Save vertices to a temporary text file
    logging.info("Saving PyVista mesh vertices to temporary file...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
        np.savetxt(temp_file.name, pv_mesh.points)
        temp_file_path = temp_file.name

    logging.info("Exiting create_mesh_with_curvature()")
    return temp_file_path, pv_mesh


##################################
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


    for i in range(1000):
        # Select a random point from the points array
        random_point = random.choice(points.tolist())
        
        # Query the K nearest neighbors for the selected point
        distances, _ = tree.query(random_point, k=K)
        
        # Exclude the distance to itself (which is always 0)
        distance = distances[1]
        
        total_distance += distance
        total_pairs += 1

    average_distance = total_distance / total_pairs if total_pairs > 0 else 0
    
    radii_list = np.linspace(0.25*average_distance,10*average_distance,10)

    return {'average_distance': average_distance,
            'total_pairs': total_pairs,
            'radii_list': radii_list}



##################################
def validate_shape(file_path):
    logging.info("Inside validate_shape()")
    temp_file_path, mesh = create_mesh_with_curvature(file_path)

    if temp_file_path:
        # Initialize PointCloud with the temporary text file
        pcl = PointCloud(temp_file_path)

        # Ensure the required steps are performed before curvature calculation
        pcl.plant_kdtree(k_neighbors=100)  # Ensure the KD-Tree is planted

        print("Running neighbor study")
        converged_neighbors_int = pcl.explicit_quadratic_neighbor_study()
        print(f"Converged Num of neighbors from explicit_quadratic_neighbor_study is {converged_neighbors_int}")
       

        print("Calculating quadratic surfaces")
        pcl.fit_explicit_quadratic_surfaces_to_neighborhoods()

        print("calculating quadratic curvatures")
        gaussian_curvature, mean_curvature = pcl.calculate_curvatures_of_explicit_quadratic_surfaces_for_all_points()
        # print(f'Gaussian first three: {gaussian_curvature[0:3]}')
        # print(f'Mean first three: {mean_curvature[0:3]}')

        print("plotting quadratic curvatures")
        pcl.plot_points_colored_by_quadratic_curvatures()

        print("saving to ply format")
        
        # Manually save the point cloud with curvature and face data to a PLY file
        points = pcl.points
        # normals = pcl.normals
        # print(f'pcl faces: {pcl.faces}')
        # faces = pcl.faces if hasattr(pcl, 'faces') and pcl.faces is not None else None
        
        # Ensure faces are in the correct format for PLY
        # if faces is not None:
        #     faces = np.hstack([[len(face)] + list(face) for face in faces])

        with open('output_with_curvatures.ply', 'w') as ply_file:
            # Write the header
            ply_file.write('ply\n')
            ply_file.write('format ascii 1.0\n')
            ply_file.write(f'element vertex {len(points)}\n')
            ply_file.write('property float x\n')
            ply_file.write('property float y\n')
            ply_file.write('property float z\n')
            # ply_file.write('property float nx\n')
            # ply_file.write('property float ny\n')
            # ply_file.write('property float nz\n')
            ply_file.write('property float gaussian_curvature\n')
            ply_file.write('property float mean_curvature\n')
            # if faces is not None:
            #     ply_file.write(f'element face {len(faces)}\n')
            #     ply_file.write('property list uchar int vertex_indices\n')
            ply_file.write('end_header\n')

            # Write the vertex data
            for i in range(len(points)):
                ply_file.write(f'{points[i][0]} {points[i][1]} {points[i][2]} '
                               f'{gaussian_curvature[i]} {mean_curvature[i]}\n')

            # # Write the face data
            # if faces is not None:
            #     for face in faces:
            #         ply_file.write(f'{face[0]} {face[1]} {face[2]} {face[3]}\n')

        print("Point cloud with curvatures saved successfully.")


        pv_mesh = mesh
        pv_mesh.point_data['gaussian_curvature'] = gaussian_curvature
        pv_mesh.point_data['mean_curvature'] = mean_curvature

        mean_curvature_squared = [item * item for item in mean_curvature]
        pv_mesh.point_data['mean_curvature_squared'] = mean_curvature_squared

        computed_bending_energy, computed_stretching_energy = load_mesh_compute_energies(pv_mesh)
        print(f'computed bending energy: {computed_bending_energy}')
        print(f'computed stretching energy: {computed_stretching_energy}')

        # Calculate mean and standard deviation for Gaussian curvature
        gaussian_mean = np.mean(gaussian_curvature)
        gaussian_std = np.std(gaussian_curvature)
        gaussian_min = gaussian_mean - 2*gaussian_std
        gaussian_max = gaussian_mean + 2*gaussian_std

        # Set color scale limits to 1 standard deviation from the mean
        gaussian_clim = [gaussian_min, gaussian_max]

        # Calculate mean and standard deviation for Mean curvature squared
        mean_curvature_squared_mean = np.mean(mean_curvature_squared)
        mean_curvature_squared_std = np.std(mean_curvature_squared)
        mean_min = mean_curvature_squared_mean - 2*mean_curvature_squared_std
        mean_max = mean_curvature_squared_mean + 2*mean_curvature_squared_std

        # Set color scale limits to 1 standard deviation from the mean
        mean_clim = [mean_min, mean_max]

        # Plot Gaussian curvature with color scale limits set to 1 std deviation from mean
        pv_mesh.plot(show_edges=False, scalars='gaussian_curvature', cmap='viridis', clim=gaussian_clim)

        # Plot Mean curvature squared with color scale limits set to 1 std deviation from mean
        pv_mesh.plot(show_edges=False, scalars='mean_curvature_squared', cmap='plasma', clim=mean_clim)



    else:
        print(f"Failed to create or load mesh.")

    logging.info("Exiting validate_shape()")

def convert_pv_to_o3d(pv_mesh):
    """
    Convert a PyVista mesh to an Open3D TriangleMesh.
    """
    # Extract vertices and faces
    vertices = np.array(pv_mesh.points)
    faces = np.array(pv_mesh.faces).reshape(-1, 4)[:, 1:]  # Reshape and remove first column

    # Create Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # Optionally add vertex normals if they exist
    if 'normals' in pv_mesh.point_data:
        o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(pv_mesh.point_data['normals'])
    
    return o3d_mesh

##################################
def load_mesh_compute_energies(mesh):
    o3d_mesh = convert_pv_to_o3d(mesh)
    logging.info("Inside load_mesh_compute_energies()")
    
    if o3d_mesh is None or len(o3d_mesh.triangles) == 0:
        logging.error("Mesh creation failed or no cells are present.")
        return 0, 0

    # Compute cell areas manually
    o3d_mesh.compute_triangle_normals()
    vertices = np.asarray(o3d_mesh.vertices)
    triangles = np.asarray(o3d_mesh.triangles)

    areas = np.zeros(len(triangles))
    
    for i, tri in enumerate(triangles):
        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        areas[i] = area

    if areas.size == 0:
        logging.error("Error: No areas computed. Mesh might not be valid.")
        return 0, 0

    face_gaussian = np.zeros(len(triangles))
    face_mean = np.zeros(len(triangles))
    face_mean_squared = np.zeros(len(triangles))

    # Assuming gaussian_curvature and mean_curvature are available as point attributes
    gaussian_curvature = np.asarray(mesh.point_data['gaussian_curvature'])
    mean_curvature = np.asarray(mesh.point_data['mean_curvature'])
    mean_squared = []
    for item in mean_curvature:
        mean_squared.append(abs(item*item))
    mean_squared = np.asarray(mean_squared)

    for i, tri in enumerate(triangles):
        verts = np.array(tri)
        face_center = np.mean(vertices[verts], axis=0)

        # Calculate the distances from each vertex to the face center
        distances = np.linalg.norm(vertices[verts] - face_center, axis=1)
        weights = distances / np.sum(distances)

        # Weighted average of curvatures based on distance
        face_gaussian[i] = np.sum(weights * gaussian_curvature[verts])
        face_mean[i] = np.sum(weights * mean_curvature[verts])
        face_mean_squared[i] = np.sum(weights * mean_squared[verts])

        if np.isnan(face_gaussian[i]) or np.isnan(face_mean[i]):
            logging.warning(f"No curvature data for triangle {i}.")

    # logging.info(f"Face Gaussian curvatures: {face_gaussian}")
    # logging.info(f"Face mean curvatures: {face_mean}")
    

    bending_energy = np.nansum(face_mean_squared * areas)
    stretching_energy = np.nansum(face_gaussian * areas)

    total_area = np.sum(areas)
    print(f'total surface area computed: {total_area}')
    
    logging.info("Exiting load_mesh_compute_energies()")

    return bending_energy, stretching_energy

##################################
def generate_pv_shapes(num_points=10000, perturbation_strength=0.0):
    def perturb_points(points, strength):
        perturbation = np.random.normal(scale=strength, size=points.shape)
        return points + perturbation

    # Evenly spaced points on a sphere (parameterized)
    def generate_sphere_points(num_points, radius=10.0):
        indices = np.arange(0, num_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / num_points)  # Properly spaced in polar angle
        theta = np.pi * (1 + np.sqrt(5)) * indices    # Golden angle method for azimuthal angle
        x = radius * np.cos(theta) * np.sin(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(phi)
        return np.vstack((x, y, z)).T

    # Evenly spaced points on a cylinder surface (parameterized)
    def generate_cylinder_points(num_points, radius=5.0, height=10.0):
        num_circumference_points = int(np.sqrt(num_points * (2 * np.pi * radius) / (2 * np.pi * radius + height)))
        num_height_points = num_points // num_circumference_points

        z = np.linspace(-height / 2, height / 2, num_height_points)
        theta = np.linspace(0, 2 * np.pi, num_circumference_points, endpoint=False)
        theta, z = np.meshgrid(theta, z)

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # Evenly spaced points on a torus surface (parameterized)
    def generate_torus_points(num_points, tube_radius=10.0, cross_section_radius=3.0):
        num_around_tube = int(np.sqrt(num_points * (2 * np.pi * tube_radius) / (2 * np.pi * tube_radius + 2 * np.pi * cross_section_radius)))
        num_around_cross_section = num_points // num_around_tube

        theta = np.linspace(0, 2 * np.pi, num_around_tube, endpoint=False)  # Angle around the tube
        phi = np.linspace(0, 2 * np.pi, num_around_cross_section, endpoint=False)  # Angle around the cross-section
        theta, phi = np.meshgrid(theta, phi)

        x = (tube_radius + cross_section_radius * np.cos(phi)) * np.cos(theta)
        y = (tube_radius + cross_section_radius * np.cos(phi)) * np.sin(theta)
        z = cross_section_radius * np.sin(phi)
        return np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # Evenly spaced points for an egg carton shape (parameterized)
    def generate_egg_carton_points(num_points):
        x = np.linspace(-3, 3, int(np.sqrt(num_points)))
        y = np.linspace(-3, 3, int(np.sqrt(num_points)))
        x, y = np.meshgrid(x, y)
        z = np.sin(x) * np.cos(y)
        return np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # Generate and perturb the shapes
    sphere_points = generate_sphere_points(num_points)
    sphere = pv.PolyData(sphere_points)
    sphere_perturbed = pv.PolyData(perturb_points(sphere_points, perturbation_strength))
    print(f'theoretical sphere surface area: {4 * 3.14159 * 10**2}')

    cylinder_points = generate_cylinder_points(num_points)
    cylinder = pv.PolyData(cylinder_points)
    cylinder_perturbed = pv.PolyData(perturb_points(cylinder_points, perturbation_strength))
    print(f'theoretical cylinder surface area: {2 * 3.14159 * 5 * 10}')

    torus_points = generate_torus_points(num_points)
    torus = pv.PolyData(torus_points)
    torus_perturbed = pv.PolyData(perturb_points(torus_points, perturbation_strength))
    print(f'theoretical torus surface area: {(2 * 3.14159 * 10) * (2 * 3.14159 * 3)}')

    egg_carton_points = generate_egg_carton_points(num_points)
    egg_carton = pv.PolyData(egg_carton_points)
    egg_carton_perturbed = pv.PolyData(perturb_points(egg_carton_points, perturbation_strength))

    return (sphere, sphere_perturbed,
            cylinder, cylinder_perturbed, torus, torus_perturbed,
            egg_carton, egg_carton_perturbed)

def save_points_to_ply(points, filename):  
    with open(filename, 'w') as f:

        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')

        np.savetxt(f, points, fmt='%.6f %.6f %.6f')

        print(f"point cloud saved in ply format as {filename}")


def parse_ply(file_path):
    try:
        with open(file_path, 'r') as file:
            # Read header
            while True:
                line = file.readline().strip()
                if line == "end_header":
                    print(f"Removed header from PLY")
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
        logging.info("Assigned points from .ply to np array")
        return np.array(points)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error parsing PLY file: {e}")

