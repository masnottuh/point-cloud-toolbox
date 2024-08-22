import numpy as np
import open3d as o3d
import pyvista as pv
import logging
import tempfile
from pointCloudToolbox import PointCloud

logging.basicConfig(level=logging.INFO)

def parse_point_cloud(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                x, y, z = map(float, parts[:3])
                points.append([x, y, z])
            except ValueError:
                continue
    return np.array(points)

def auto_determine_bpa_radii(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_distance = np.mean(distances)
    radii = [avg_distance * factor for factor in [10, 50, 250, 400, 800]]
    return radii

def downsample_point_cloud(points, num_points):
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    return points

def create_mesh_with_curvature(file_path, k_neighbors, shape, num_points=None):
    points = parse_point_cloud(file_path)
    if points is None:
        return None, None, None

    # Downsample point cloud if specified
    if num_points is not None:
        points = downsample_point_cloud(points, num_points)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals
    logging.info("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.normalize_normals()

    # Remove statistical outliers
    logging.info("Removing outliers...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)

    # Determine BPA radii based on point cloud density
    radii = auto_determine_bpa_radii(points)
    logging.info(f"Initial BPA radii: {radii}")

    # Perform BPA triangulation using Open3D and dynamically adjust radii
    logging.info("Using Ball-Pivoting Algorithm (BPA) for reconstruction...")
    bpa_mesh = None
    for radius in radii:
        bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius]))
        num_holes = len(bpa_mesh.get_non_manifold_edges())
        logging.info(f"BPA with radius {radius}: Number of holes = {num_holes}")
        if num_holes == 0:
            break

    logging.info("BPA triangulation completed. Number of vertices: %d", len(bpa_mesh.vertices))

    if len(bpa_mesh.triangles) == 0:
        logging.error("Error: BPA triangulation failed to create any triangles.")
        return None, None, None

    # Convert the Open3D mesh to PyVista for visualization and further processing
    pv_mesh = pv.PolyData(np.asarray(bpa_mesh.vertices))
    faces = np.hstack([[3, *triangle] for triangle in np.asarray(bpa_mesh.triangles)])
    pv_mesh.faces = faces
    pv_mesh.point_data["normals"] = np.asarray(bpa_mesh.vertex_normals)

    # Clean the mesh using PyVista
    pv_mesh.clean(inplace=True)
    pv_mesh.fill_holes(hole_size=2.0, inplace=True)

    # Save the vertices to a temporary text file that PointCloud can read
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
        np.savetxt(temp_file.name, pv_mesh.points)
        temp_file_path = temp_file.name

    return temp_file_path, pv_mesh, ind

def compute_face_areas(mesh):
    logging.info(f"Mesh has {mesh.n_faces} faces")

    # Compute areas of the triangular faces
    faces = mesh.faces.reshape((-1, 4))[:, 1:4]
    points = mesh.points
    areas = np.zeros(faces.shape[0])

    valid_faces = []

    for i, face in enumerate(faces):
        p1, p2, p3 = points[face]
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        if area > 0:
            areas[i] = area
            valid_faces.append(i)

    logging.info(f"Computed {len(valid_faces)} valid face areas")
    return areas, valid_faces

def load_mesh_compute_energies(mesh):
    if mesh is None:
        logging.error("Mesh creation failed or no cells are present.")
        return 0, 0

    # Compute face areas
    areas, valid_faces = compute_face_areas(mesh)
    if len(valid_faces) == 0:
        logging.error("Error: No valid areas computed.")
        return 0, 0
    logging.info(f"Computed areas: {areas}")

    # Output total area of the mesh
    total_area = np.sum(areas)
    logging.info(f"Total area of the mesh: {total_area:.6f}")

    face_gaussian = np.full(mesh.n_faces, np.nan)
    face_mean = np.full(mesh.n_faces, np.nan)

    if mesh.faces.size == 0:
        logging.error("Error: No faces found in mesh.")
        return 0, 0

    for i in valid_faces:
        verts = mesh.faces[1 + i * 4: 4 + i * 4]
        gauss_curv = mesh.point_data['gaussian_curvature'][verts]
        mean_curv = mesh.point_data['mean_curvature'][verts]
        face_gaussian[i] = np.mean(gauss_curv)
        face_mean[i] = np.mean(mean_curv)

    valid_indices = ~np.isnan(face_gaussian) & ~np.isnan(face_mean)
    areas = areas[valid_indices]
    face_gaussian = face_gaussian[valid_indices]
    face_mean = face_mean[valid_indices]

    logging.info(f"Face Gaussian curvatures: {face_gaussian}")
    logging.info(f"Face mean curvatures: {face_mean}")

    # Visualize curvature values
    mesh.point_data['gaussian_curvature'] = np.full(mesh.n_points, np.nan)
    mesh.point_data['mean_curvature_squared'] = np.full(mesh.n_points, np.nan)
    mesh.point_data['gaussian_curvature'][valid_indices] = face_gaussian
    mesh.point_data['mean_curvature_squared'][valid_indices] = face_mean ** 2

    pv.plot(mesh, scalars='gaussian_curvature', title='Gaussian Curvature')
    pv.plot(mesh, scalars='mean_curvature_squared', title='Mean Curvature Squared')

    bending_energy = np.sum(face_mean ** 2 * areas)
    stretching_energy = np.sum(face_gaussian * areas)

    # Visualize energy values
    mesh.point_data['stretching_energy'] = np.zeros(mesh.n_points)
    mesh.point_data['bending_energy'] = np.zeros(mesh.n_points)
    mesh.point_data['stretching_energy'][valid_indices] = face_gaussian * areas
    mesh.point_data['bending_energy'][valid_indices] = face_mean ** 2 * areas

    pv.plot(mesh, scalars='stretching_energy', title='Stretching Energy')
    pv.plot(mesh, scalars='bending_energy', title='Bending Energy')

    return bending_energy, stretching_energy

def validate_shape(shape, theoretical_bending_energy, theoretical_stretching_energy, file_path, k_neighbors, num_points=None):
    temp_file_path, pv_mesh, valid_indexes = create_mesh_with_curvature(file_path, k_neighbors, shape, num_points)
    if temp_file_path:
        # Initialize PointCloud with the temporary text file
        pcl = PointCloud(temp_file_path, k_neighbors=k_neighbors)

        # Ensure the required steps are performed before curvature calculation
        pcl.plant_kdtree(k_neighbors=k_neighbors)  # Ensure the KD-Tree is planted
        pcl.visualize_knn_for_n_random_points(5, k_neighbors)  # Initialize random_points and random_indexes

        print("Running neighbor study")
        pcl.explicit_quadratic_neighbor_study()  # use 'goldmans' or 'standard'

        print("Calculating quadratic surfaces")
        pcl.fit_explicit_quadratic_surfaces_to_neighborhoods()

        print("calculating quadratic curvatures")
        gaussian_curvature, mean_curvature = pcl.calculate_curvatures_of_explicit_quadratic_surfaces_for_all_points()  # use 'goldmans' or 'standard'

        print("plotting quadratic curvatures")
        pcl.plot_points_colored_by_quadratic_curvatures()

        print("saving to ply format")
        mesh_path = "output_with_curvatures_and_normals.ply"
        pcl.export_ply_with_curvature_and_normals(mesh_path)

        # Convert lists to numpy arrays
        gaussian_curvature = np.array(gaussian_curvature)
        mean_curvature = np.array(mean_curvature)

        if gaussian_curvature.size == 0 or mean_curvature.size == 0:
            logging.error("Error: Curvature calculation failed or returned empty arrays.")
            return

        pv_mesh.point_data['gaussian_curvature'][valid_indexes] = gaussian_curvature
        pv_mesh.point_data['mean_curvature'][valid_indexes] = mean_curvature

        computed_bending_energy, computed_stretching_energy = load_mesh_compute_energies(pv_mesh)
        print(f"{shape} - Bending Energy: Theoretical={theoretical_bending_energy:.12f}, Computed={computed_bending_energy:.12f}")
        if theoretical_stretching_energy is not None:
            print(f"{shape} - Stretching Energy: Theoretical={theoretical_stretching_energy:.12f}, Computed={computed_stretching_energy:.12f}")
        else:
            print(f"{shape} - Stretching Energy: Computed={computed_stretching_energy:.12f}")
    else:
        print(f"Failed to create or load mesh for {shape}.")


def generate_torus(num_points, tube_radius, cross_section_radius):
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi = np.linspace(0, 2 * np.pi, num_points)
    theta, phi = np.meshgrid(theta, phi)
    theta, phi = theta.flatten(), phi.flatten()
    x = (tube_radius + cross_section_radius * np.cos(phi)) * np.cos(theta)
    y = (tube_radius + cross_section_radius * np.cos(phi)) * np.sin(theta)
    z = cross_section_radius * np.sin(phi)
    points = np.vstack((x, y, z)).T
    return points

def generate_bumpy_plane(num_points, width, length, bump_height):
    x = np.linspace(-width / 2, width / 2, int(np.sqrt(num_points)))
    y = np.linspace(-length / 2, length / 2, int(np.sqrt(num_points)))
    x, y = np.meshgrid(x, y)
    z = bump_height * np.sin(np.pi * x / width) * np.cos(np.pi * y / length)
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    return points

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

# Generate and save a torus
torus_points = generate_torus(num_points=10000, tube_radius=10, cross_section_radius=3)
save_points_to_ply(torus_points, 'torus.ply')

# Generate and save a bumpy plane
bumpy_plane_points = generate_bumpy_plane(num_points=10000, width=20, length=20, bump_height=2)
save_points_to_ply(bumpy_plane_points, 'bumpy_plane.ply')

validate_shape("1", 0, 0, '1_UState.asc', 85, num_points=10000)
validate_shape("2", 0, 0, '2_ValleyPropogation.asc', 85, num_points=10000)
validate_shape("3", 0, 0, '3_SRidgesProp.asc', 85, num_points=10000)
validate_shape("4", 0, 0, '4_SRidgeConverging.asc', 85, num_points=10000)
validate_shape("5", 0, 0, '5_MoreConverging.asc', 85, num_points=10000)
validate_shape("6", 0, 0, '6_FinalSRidge.asc', 85, num_points=10000)
validate_shape("7", 0, 0, '7_MState.asc', 85, num_points=10000)
validate_shape("8", 0, 0, '8.asc', 85, num_points=10000)
validate_shape("9", 0, 0, '9.asc', 85, num_points=10000)
validate_shape("10", 0, 0, '10.asc', 85, num_points=10000)
validate_shape("11", 0, 0, '11_Final.asc', 85, num_points=10000)
