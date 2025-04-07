import pyvista as pv
import numpy as np


# Load mean curvature data
mesh = np.load('NN_cylinder_none_radius_None_points_49465_mean.npy')



# Visualize the mesh
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='Mean Curvature', cmap='coolwarm', show_scalar_bar=True)
plotter.add_scalar_bar(title='Mean Curvature')
plotter.show()
