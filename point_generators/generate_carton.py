import numpy as np

# Define the number of points n
n = 100000  # Example value; can be changed as needed

# Define the bumpy "egg-carton" surface function
def egg_carton_surface(x, y):
    return np.sin(x) * np.sin(y)

# Generate random (x, y) coordinates within a range that will produce a nice egg-carton pattern
xy_range = np.linspace(-2*np.pi, 2*np.pi, int(np.sqrt(n)))
x, y = np.meshgrid(xy_range, xy_range)
x = x.flatten()
y = y.flatten()

# Calculate the corresponding z values for the surface
z = egg_carton_surface(x, y)

# Create the point cloud
points = np.column_stack((x, y, z))

# Save the point cloud to a text file
file_path = './sample_scans/egg_carton.txt'
np.savetxt(file_path, points, fmt='%.6f')
