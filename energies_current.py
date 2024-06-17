import numpy as np
import pyvista as pv

def parse_ply(file_path):
    with open(file_path, 'r') as file:
        # Read header
        while True:
            line = file.readline().strip()
            if line == "end_header":
                break
        # Read body data
        points = []
        gaussian_curvature = []
        mean_curvature = []
        while True:
            line = file.readline()
            if not line:
                break
            parts = line.split()
            x, y, z = map(float, parts[:3])
            gauss, mean = map(float, parts[3:])
            points.append([x, y, z])
            gaussian_curvature.append(gauss)
            mean_curvature.append(mean)
    return np.array(points), np.array(gaussian_curvature), np.array(mean_curvature)

def create_mesh_with_curvature(file_path):
    points, gauss_curv, mean_curv = parse_ply(file_path)
    mesh = pv.PolyData(points)
    if mesh.n_points > 0:
        print("Attempting Delaunay 3D triangulation...")
        mesh = mesh.delaunay_3d()
        print("Triangulation completed. Number of cells:", mesh.n_cells)
        
        # Extract the outer surface of the Delaunay volume
        surface = mesh.extract_surface()
        print("Extracted surface. Number of cells:", surface.n_cells)

        if surface.n_cells == 0:
            print("No surface cells were generated.")
            return None
        
        surface.point_data['gaussian_curvature'] = gauss_curv
        surface.point_data['mean_curvature'] = mean_curv
        return surface
    return None

def load_mesh_compute_energies(mesh):
    if mesh is None:
        print("Mesh creation failed or no cells are present.")
        return 0, 0
    mesh = mesh.triangulate()  # Ensure the mesh is triangulated

    areas = mesh.compute_cell_sizes(length=False, area=True, volume=False).cell_data['Area']
    face_gaussian = np.zeros(mesh.n_cells)
    face_mean = np.zeros(mesh.n_cells)

    for i in range(mesh.n_cells):
        verts = mesh.faces[1 + i * 4: 4 + i * 4]
        gauss_curv = mesh.point_data['gaussian_curvature'][verts]
        mean_curv = mesh.point_data['mean_curvature'][verts]
        face_gaussian[i] = np.mean(gauss_curv)
        face_mean[i] = np.mean(mean_curv)

    bending_energy = np.sum(face_mean ** 2 * areas)
    stretching_energy = np.sum(face_gaussian * areas)

    return bending_energy, stretching_energy

# Usage
file_path = 'output_with_curvatures.ply'
mesh = create_mesh_with_curvature(file_path)
if mesh:
    bending_energy, stretching_energy = load_mesh_compute_energies(mesh)
    print(f"Bending Energy: {bending_energy:.12f}")
    print(f"Stretching Energy: {stretching_energy:.12f}")
    mesh.plot(show_edges=True)
else:
    print("Failed to create or load mesh.")
