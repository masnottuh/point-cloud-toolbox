import numpy as np
def generate_mobius_strip(num_points=1000, length=2, width=1):
    # Parametric equations for a Möbius strip
    def mobius_strip(u, v):
        # u varies from 0 to 2*pi, v varies from -width/2 to +width/2
        half = (0 <= u) & (u < np.pi)
        x = (1 + v / 2 * np.cos(u / 2)) * np.cos(u)
        y = (1 + v / 2 * np.cos(u / 2)) * np.sin(u)
        z = v / 2 * np.sin(u / 2)
        return x, y, z

    # Generate u and v values
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(-width / 2, width / 2, num_points)
    u, v = np.meshgrid(u, v)

    # Generate the points on the surface of the Möbius strip
    x, y, z = mobius_strip(u.flatten(), v.flatten())

    # Combine x, y, and z into a single array
    mobius_points = np.column_stack((x, y, z))

    return mobius_points

# Generate the point cloud for the Möbius strip
mobius_points = generate_mobius_strip()

# Save the points to a text file
mobius_points_path = './sample_scans/mobius_strip.txt'
np.savetxt(mobius_points_path, mobius_points, fmt='%.6f')

