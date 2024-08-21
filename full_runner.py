import numpy as np
import open3d as o3d
import pyvista as pv
import logging
import tempfile
from pointCloudToolbox import PointCloud

logging.basicConfig(level=logging.INFO)

def parse_ply(file_path):
    try:
        with open(file_path, 'r') as file:
            # Read header
            while True:
                line = file.readline().strip()
                if line == "end_header":
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
        return None

def create_mesh_with_curvature(file_path, k_neighbors):
    points = parse_ply(file_path)
    if points is None:
        return None

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

    # Perform Ball-Pivoting Algorithm (BPA) for mesh reconstruction
    logging.info("Using Ball-Pivoting Algorithm (BPA) for reconstruction...")
    radii = [0.05, 0.1, 0.2]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    logging.info("BPA reconstruction completed. Number of vertices: %d", len(mesh.vertices))

    # Convert Open3D mesh to PyVista mesh
    pv_mesh = pv.PolyData(np.asarray(mesh.vertices), np.hstack([[3] + face.tolist() for face in np.asarray(mesh.triangles)]))

    # Save the vertices to a temporary text file that PointCloud can read
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
        np.savetxt(temp_file.name, pv_mesh.points)
        temp_file_path = temp_file.name

    return temp_file_path

def load_mesh_compute_energies(mesh):
    if mesh is None:
        logging.error("Mesh creation failed or no cells are present.")
        return 0, 0
    mesh = mesh.triangulate()  # Ensure the mesh is triangulated

    areas = mesh.compute_cell_sizes(length=False, area=True, volume=False).cell_data['Area']
    if areas.size == 0:
        logging.error("Error: No areas computed.")
        return 0, 0
    logging.info(f"Computed areas: {areas}")

    face_gaussian = np.zeros(mesh.n_cells)
    face_mean = np.zeros(mesh.n_cells)

    if mesh.faces.size == 0:
        logging.error("Error: No faces found in mesh.")
        return 0, 0

    for i in range(mesh.n_cells):
        verts = mesh.faces[1 + i * 4: 4 + i * 4]
        gauss_curv = mesh.point_data['gaussian_curvature'][verts]
        mean_curv = mesh.point_data['mean_curvature'][verts]
        face_gaussian[i] = np.mean(gauss_curv)
        face_mean[i] = np.mean(mean_curv)

    logging.info(f"Face Gaussian curvatures: {face_gaussian}")
    logging.info(f"Face mean curvatures: {face_mean}")

    bending_energy = np.sum(face_mean ** 2 * areas)
    stretching_energy = np.sum(face_gaussian * areas)

    return bending_energy, stretching_energy

def validate_shape(shape, theoretical_bending_energy, theoretical_stretching_energy, file_path, k_neighbors):
    temp_file_path = create_mesh_with_curvature(file_path, k_neighbors)
    if temp_file_path:
        # Initialize PointCloud with the temporary text file
        pcl = PointCloud(temp_file_path, k_neighbors=k_neighbors)

        # Ensure the required steps are performed before curvature calculation
        pcl.plant_kdtree(k_neighbors=k_neighbors)  # Ensure the KD-Tree is planted

        print("Running neighbor study")
        pcl.explicit_quadratic_neighbor_study()  # use 'goldmans' or 'standard'

        print("Calculating quadratic surfaces")
        pcl.fit_explicit_quadratic_surfaces_to_neighborhoods()

        print("calculating quadratic curvatures")
        gaussian_curvature, mean_curvature = pcl.calculate_curvatures_of_explicit_quadratic_surfaces_for_all_points()  # use 'goldmans' or 'standard'

        print("plotting quadratic curvatures")
        pcl.plot_points_colored_by_quadratic_curvatures()

        print("saving to ply format")
        mesh_path = "output_with_curvatures.ply"
        pcl.export_ply_with_curvature_and_normals('output_with_curvatures_and_normals.ply')

        # if not gaussian_curvature.size or not mean_curvature.size:
        #     logging.error("Error: Curvature calculation failed or returned empty arrays.")
        #     return

        # Convert the points back to a PyVista mesh for further processing
        pv_mesh = pv.PolyData(np.loadtxt(temp_file_path))
        pv_mesh.point_data['gaussian_curvature'] = gaussian_curvature
        pv_mesh.point_data['mean_curvature'] = mean_curvature

        computed_bending_energy, computed_stretching_energy = load_mesh_compute_energies(pv_mesh)
        print(f"{shape} - Bending Energy: Theoretical={theoretical_bending_energy:.12f}, Computed={computed_bending_energy:.12f}")
        if theoretical_stretching_energy is not None:
            print(f"{shape} - Stretching Energy: Theoretical={theoretical_stretching_energy:.12f}, Computed={computed_stretching_energy:.12f}")
        else:
            print(f"{shape} - Stretching Energy: Computed={computed_stretching_energy:.12f}")
        pv_mesh.plot(show_edges=True)
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

# Validate known shapes with appropriate k_neighbors for surface fitting
validate_shape("Sphere", 4 * np.pi, 4 * np.pi, 'output_with_curvatures_sphere.ply', 125)
# validate_shape("Cylinder", np.pi * 20 / (2 * 10), 0, 'output_with_curvatures_cylinder.ply', 825)  # Adjust height and radius accordingly
# validate_shape("Plane", 0, 0, 'output_with_curvatures_plane.ply', 825)
# validate_shape("Saddle", 0, None, 'output_with_curvatures_saddle.ply', 825)  # Replace None with the correct value if known
# validate_shape("Torus", 2 * np.pi**2 * 10 * 3, 0, 'torus.ply', 825)
# validate_shape("Bumpy Plane", 0, 0, 'bumpy_plane.ply', 825)
# validate_shape("Custom SRidge", 0, 0, 'sample_scans/sridge.txt', 825)
