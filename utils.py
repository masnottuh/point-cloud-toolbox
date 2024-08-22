import pyvista as pv
import numpy as np
import tempfile
import open3d as o3d
import logging
from scipy.spatial import KDTree
import random
from pointCloudToolbox import *

def create_mesh_with_curvature(file_path):
        logging.info("Inside create_mesh_with_curvature()")
        points = parse_ply(file_path)
        if points is None:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        mets = average_distance_using_kd_tree(pcd)
        radii_list = mets['radii_list']
        # Estimate normals
        logging.info("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radii_list[0], len(pcd.points)//100))
        pcd.normalize_normals()

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
        pv_mesh = pv.PolyData(np.asarray(mesh.vertices), np.hstack([[3] + face.tolist() for face in np.asarray(mesh.triangles)]))
            
        # Save the vertices to a temporary text file that PointCloud can read
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
            np.savetxt(temp_file.name, pv_mesh.points)
            temp_file_path = temp_file.name
        logging.info("exiting create_mesh_with_curvature()")
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

    for i in range(500):
        # Query the K nearest neighbors, including the point itself
        distances, _ = tree.query(random.choice(points), k=K)
        # Exclude the distance to itself (which is always 0)
        distance = distances[1]
        # print(distance)
        
        total_distance += distance
        total_pairs += 1

    average_distance = total_distance / total_pairs if total_pairs > 0 else 0
    
    radii_list = np.linspace(average_distance,10*average_distance,3)

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
        pcl.explicit_quadratic_neighbor_study()

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


        # Load the PLY file using pyvista
        pv_mesh = mesh
        pv_mesh.point_data['gaussian_curvature'] = gaussian_curvature
        pv_mesh.point_data['mean_curvature'] = mean_curvature

        mean_curvature_squared = []
        for item in mean_curvature:
            mean_curvature_squared.append(item*item)

        pv_mesh.point_data['mean_curvature_squared'] = mean_curvature_squared

        computed_bending_energy, computed_stretching_energy = load_mesh_compute_energies(pv_mesh)
        print(f'computed bending energy: {computed_bending_energy}')
        print(f'computed stretching energy: {computed_stretching_energy}')
        
    
        # Calculate the range for Gaussian curvature
        gaussian_min = pv_mesh.point_data['gaussian_curvature'].min()
        gaussian_max = pv_mesh.point_data['gaussian_curvature'].max()
        gaussian_range = gaussian_max - gaussian_min

        # Set the color scale limits for Gaussian curvature only if the range is less than 0.001
        if gaussian_range < 0.001:
            gaussian_clim = [gaussian_min, gaussian_max]
        else:
            gaussian_clim = None  # Let PyVista determine the color scale automatically

        # Calculate the range for Mean curvature squared
        mean_min = pv_mesh.point_data['mean_curvature_squared'].min()
        mean_max = pv_mesh.point_data['mean_curvature_squared'].max()
        mean_range = mean_max - mean_min

        # Set the color scale limits for Mean curvature squared only if the range is less than 0.001
        if mean_range < 0.01:
            mean_clim = [mean_min, mean_max]
        else:
            mean_clim = None  # Let PyVista determine the color scale automatically

        # Plot Gaussian curvature with conditional color scale limits
        pv_mesh.plot(show_edges=False, scalars='gaussian_curvature', cmap='viridis', clim=gaussian_clim)

        # Plot Mean curvature squared with conditional color scale limits
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
def generate_pv_shapes(num_points=10000, perturbation_strength=0.05):
    def perturb_points(points, strength):
        perturbation = np.random.normal(scale=strength, size=points.shape)
        return points + perturbation

    # Create a plane
    plane = pv.Plane(center=(0, 0, 0), direction=(0, 10, 10), i_size=10, j_size=10)
    plane_perturbed = pv.Plane(center=(0, 0, 0), direction=(0, 10, 10), i_size=10, j_size=10)
    plane_perturbed.points = perturb_points(plane_perturbed.points, perturbation_strength)
    
    # Create a sphere manually
    radius = 10.0
    phi = np.linspace(0, np.pi, int(np.sqrt(num_points)))
    theta = np.linspace(0, 2 * np.pi, int(np.sqrt(num_points)))
    phi, theta = np.meshgrid(phi, theta)
    phi, theta = phi.flatten(), theta.flatten()

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    sphere_points = np.vstack((x, y, z)).T
    sphere = pv.PolyData(sphere_points)
    sphere_perturbed = pv.PolyData(perturb_points(sphere_points, perturbation_strength))
    print(f'theoretical sphere surface area: {4*3.14159*radius*radius}')

    # Create a cylinder manually with points on the curved wall and flat caps
    radius2 = 5
    height2 = 10
    resolution2 = 200

    # Curved wall of the cylinder
    theta = np.linspace(0, 2 * np.pi, resolution2)
    z = np.linspace(-height2 / 2, height2 / 2, resolution2)
    theta, z = np.meshgrid(theta, z)
    x = radius2 * np.cos(theta)
    y = radius2 * np.sin(theta)
    
    cylinder_wall_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    cylinder_wall = pv.PolyData(cylinder_wall_points)
    cylinder_perturbed = pv.PolyData(perturb_points(cylinder_wall_points, perturbation_strength))

    cylinder = cylinder_wall
    print(f'theoretical cylinder surface area: {2*3.14159*radius2*height2}')

    # Create a torus using the provided method for generating points
    tube_radius = 10  # Major radius (tube radius)
    cross_section_radius = 3  # Minor radius (cross-section radius)
    print(f'theoretical torus surface area: {(2*3.14159*tube_radius)*(2*3.14159*cross_section_radius)}')
    theta = np.linspace(0, 2 * np.pi, int(np.sqrt(num_points)))
    phi = np.linspace(0, 2 * np.pi, int(np.sqrt(num_points)))
    theta, phi = np.meshgrid(theta, phi)
    theta, phi = theta.flatten(), phi.flatten()

    x = (tube_radius + cross_section_radius * np.cos(phi)) * np.cos(theta)
    y = (tube_radius + cross_section_radius * np.cos(phi)) * np.sin(theta)
    z = cross_section_radius * np.sin(phi)

    torus_points = np.vstack((x, y, z)).T
    torus = pv.PolyData(torus_points)
    torus_perturbed = pv.PolyData(perturb_points(torus_points, perturbation_strength))
    
    # Create an egg carton shape
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    x, y = np.meshgrid(x, y)
    z = np.sin(x) * np.cos(y)
    egg_carton = pv.StructuredGrid(x, y, z)
    egg_carton_perturbed = pv.StructuredGrid(x, y, perturb_points(z, perturbation_strength))
    
    return (plane, plane_perturbed, sphere, sphere_perturbed,
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
        return np.array(points)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error parsing PLY file: {e}")