import os
import pyvista as pv

def open_mesh_in_separate_window(mesh_path):
    try:
        mesh = pv.read(mesh_path)
        # Create an interactive plotter window.
        plotter = pv.Plotter(window_size=[800, 600])
        plotter.add_mesh(mesh, show_edges=True, color="lightblue")
        # Use add_text() to display a title (or filename) since the plot() method doesn't accept a title keyword.
        plotter.add_text(os.path.basename(mesh_path), position="upper_left", font_size=10)
        # Display the window interactively.
        plotter.show()
    except Exception as e:
        print(f"Failed to load {mesh_path}: {e}")

# Directory where the meshes are saved.
mesh_dir = "mesh_snaps"
# Valid mesh file extensions.
valid_ext = ('.vtk', '.vtp', '.ply')

# Gather full paths for valid mesh files.
mesh_files = [os.path.join(mesh_dir, f) for f in os.listdir(mesh_dir) if f.lower().endswith(valid_ext)]

# Open each mesh in a separate interactive window sequentially.
for mesh_file in mesh_files:
    print(f"Opening {mesh_file}...")
    open_mesh_in_separate_window(mesh_file)
