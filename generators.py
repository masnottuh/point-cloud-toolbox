import numpy as np

def generate_torus_points(R, r, num_points):
    """
    Generate points on the surface of a torus.

    Parameters:
    - R: Major radius of the torus (distance from the center of the tube to the center of the torus)
    - r: Minor radius of the torus (radius of the tube)
    - num_points: Number of points to generate

    Returns:
    - points: Numpy array of points on the surface of the torus
    """
    # Initialize an array to hold the points
    points = []

    # Generate points
    for _ in range(num_points):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta)
        points.append([x, y, z])

    return np.array(points)

def generate_monkey_saddle_points(a, b, num_points):
    """
    Generate points on the surface of a monkey saddle.
    
    Parameters:
    - a: Range for x and y (from -a to a)
    - b: Coefficient for the monkey saddle equation z = b * (x^3 - 3xy^2)
    - num_points: Number of points to generate
    
    Returns:
    - points: Numpy array of points on the surface of the monkey saddle
    """
    points = []
    
    # Generate points
    for _ in range(num_points):
        x = np.random.uniform(-a, a)
        y = np.random.uniform(-a, a)
        z = b * (x**3 - 3*x*y**2)
        points.append([x, y, z])
    
    return np.array(points)

def generate_bumpy_spheroid_points(a, c, num_points, bumpiness):
    """
    Generate points on the surface of a bumpy spheroid.
    
    Parameters:
    - a: Semi-major axis of the spheroid (equatorial radius)
    - c: Semi-minor axis of the spheroid (polar radius)
    - num_points: Number of points to generate
    - bumpiness: Amplitude of the sinusoidal bumpiness
    
    Returns:
    - points: Numpy array of points on the surface of the bumpy spheroid
    """
    points = []
    
    # Generate points
    for _ in range(num_points):
        theta = np.random.uniform(0, np.pi)  # Elevation angle
        phi = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
        r = a * (1 + bumpiness * np.sin(4 * phi) * np.sin(4 * theta))  # Perturb the radius for bumpiness
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = c * np.cos(theta)
        points.append([x, y, z])
    
    return np.array(points)

# Define the parametric equations for the Klein Bottle
def klein_bottle(u, v):
    half = (0 <= u) & (u < np.pi)
    r = 4 * (1 - np.cos(u) / 2)
    x = 6 * np.cos(u) * (1 + np.sin(u)) + r * np.cos(v + np.pi)
    y = 16 * np.sin(u) + r * np.sin(v + np.pi)
    z = 4 * (1 - np.cos(u) / 2) * np.sin(v)
    x[half] = 6 * np.cos(u[half]) * (1 + np.sin(u[half])) + r[half] * np.cos(u[half]) * np.cos(v[half])
    y[half] = 16 * np.sin(u[half]) + r[half] * np.sin(u[half]) * np.cos(v[half])
    z[half] = r[half] * np.sin(v[half])
    return x, y, z

# Define the parametric equations for the Dupin Cyclide
def dupin_cyclide(u, v, a=1, b=1):
    cosu = np.cos(u)
    sinu = np.sin(u)
    cosv = np.cos(v)
    sinv = np.sin(v)
    denom = a + b * cosv * cosu

    x = cosv * (a * cosu + b) / denom
    y = sinv * (a * cosu + b) / denom
    z = sinu * cosv / denom
    return x, y, z

def save_points_to_txt(file_path, points):
    """
    Save points to a txt file.

    Parameters:
    - file_path: Path to the output txt file
    - points: Numpy array of points to save
    """
    # Save the points to the file
    np.savetxt(file_path, points, fmt='%.6f')




# Parameters for the torus
major_radius = 10  # Major radius
minor_radius = 1.5   # Minor radius
n_points = 250000    # Number of points
# Generate points on the surface of the torus
torus_points = generate_torus_points(major_radius, minor_radius, n_points)
# Path for the output file
output_file_path = './sample_scans/torus.txt'
# Save the points to a txt file
save_points_to_txt(output_file_path, torus_points)




# Parameters for the monkey saddle
a_range = 2  # Range for x and y
b_coefficient = 1  # Coefficient for the monkey saddle equation
n_points = 250000   # Number of points
# Generate points on the surface of the monkey saddle
monkey_saddle_points = generate_monkey_saddle_points(a_range, b_coefficient, n_points)
# Path for the output file
output_file_path_monkey_saddle = './sample_scans/monkey_saddle.txt'
save_points_to_txt(output_file_path_monkey_saddle, monkey_saddle_points)
# Provide the path for download
output_file_path_monkey_saddle



# Parameters for the bumpy spheroid
a_radius = 5  # Semi-major axis (equatorial radius)
c_radius = 5  # Semi-minor axis (polar radius)
n_points = 250000   # Number of points
bump_amplitude = 0.1  # Amplitude of the bumpiness
# Generate points on the surface of the bumpy spheroid
bumpy_spheroid_points = generate_bumpy_spheroid_points(a_radius, c_radius, n_points, bump_amplitude)
# Path for the output file
output_file_path_bumpy_spheroid = './sample_scans/bumpy_spheroid.txt'
# Save the points to a txt file
save_points_to_txt(output_file_path_bumpy_spheroid, bumpy_spheroid_points)
# Provide the path for download
output_file_path_bumpy_spheroid


#KLEIN BOTTLE
# Create a meshgrid for the parameters u and v
u = np.linspace(0, 2 * np.pi, 750)
v = np.linspace(0, 2 * np.pi, 750)
u, v = np.meshgrid(u, v)
# Generate the points on the surface of the Klein Bottle
x, y, z = klein_bottle(u.flatten(), v.flatten())
# Combine x, y, and z into a single array
klein_points = np.column_stack((x, y, z))
# Save the points to a text file
klein_points_path = './sample_scans/klein_bottle.txt'
np.savetxt(klein_points_path, klein_points, fmt='%.6f')



#DUPIN CYCLIDE
# Create a meshgrid for the parameters u and v
u = np.linspace(0, 2 * np.pi, 750)
v = np.linspace(0, 2 * np.pi, 750)
u, v = np.meshgrid(u, v)
# Generate the points on the surface of the Dupin Cyclide
x, y, z = dupin_cyclide(u.flatten(), v.flatten())
# Combine x, y, and z into a single array
dupin_points = np.column_stack((x, y, z))
# Save the points to a text file
dupin_points_path = './sample_scans/dupin_cyclide.txt'
np.savetxt(dupin_points_path, dupin_points, fmt='%.6f')