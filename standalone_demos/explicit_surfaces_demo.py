import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares

def get_best_fit_plane(points):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(x, y, z, c='b')

    centroid = np.mean(points, axis=0)

    # centered_points = points - centroid
    centered_points = points

    Cov = np.cov(centered_points, rowvar=False)

    U, S, Vt = svd(Cov, full_matrices=True)

    normal = Vt[-1]

    ax.quiver(0,0,0,Vt[:,0], Vt[:,1], Vt[:,2],color='g', arrow_length_ratio=0.001)
    # plt.show()

    vec1 = normal
    vec2 = np.array([0, 0, 1])
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    # Rotate the points and the plane normal
    rotated_points = np.dot(rotation_matrix, points.T).T
    rotated_normal = np.dot(rotation_matrix, normal)
    ax.quiver(0,0,0,rotated_normal[0], rotated_normal[1], rotated_normal[2], color='r', arrow_length_ratio=0.001)
    # plot 
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(rotated_points[:,0], rotated_points[:,1], rotated_points[:,2], c='r')
    ax.quiver(0,0,0,rotated_normal[0], rotated_normal[1], rotated_normal[2], color='g', arrow_length_ratio=0.001)

    fit_quadratic_surface_to_points(rotated_points, ax)

def fit_quadratic_surface_to_points(points, ax):

    a = points[:,0]
    b = points[:,1]
    c = points[:,2]

    def quadratic_surface(params, a, b):
        A, B, C, D, E, F = params
        return A*a**2 + B*b**2 + C*a*b + D*a + E*b + F

    # Define the objective function for least squares (residuals)
    def objective_function(params, a, b, c):
        return quadratic_surface(params, a, b) - c

    # Initial guess for the parameters
    initial_guess = np.ones(6)

    # Perform least squares optimization
    result = least_squares(objective_function, initial_guess, args=(a, b, c))

    # The optimal parameters
    params_optimal = result.x

    # Unpack the optimal parameters
    A, B, D, E, F, G = params_optimal

    # Create a grid to plot the surface
    xx, yy = np.meshgrid(np.linspace(min_x, max_x, 50), np.linspace(min_x, max_x, 50))
    zz = quadratic_surface(params_optimal, xx, yy)

    # Plot the surface
    ax.plot_surface(xx, yy, zz, color='r', alpha=0.5, rstride=1, cstride=1, edgecolor='none')

    # Plot the data points
    ax.scatter(points[:,0], points[:,1], points[:,2], color='b')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Fitted Quadratic Surface and Original Data Points')

    # Show the plot
    plt.show()

############################################################################################################
############################################################################################################
############################################################################################################

min_x = -1
max_x = 1
min_y = -1
max_y = 1
num_points = 20
# Create a grid of x and y values
xx, yy = np.meshgrid(range(min_x, max_x), range(min_y, max_y))

# plane at z=0, tilted
x = np.random.uniform(min_x, max_x, num_points)
y = np.random.uniform(min_y, max_y, num_points)
z = 0.1*x + 0.1*y
points = np.column_stack((x, y, z))
get_best_fit_plane(points)


#tilted parabaloid
z = (0.1*x)**2 + (0.1*y)**2 + 0.1*x
points = np.column_stack((x, y, z))
get_best_fit_plane(points)


#tilted saddle
z = x**2 - y**2
points = np.column_stack((x, y, z))
get_best_fit_plane(points)


#tilted monkey saddle
z = x**3 - 3*x*y**2
points = np.column_stack((x, y, z))
get_best_fit_plane(points)


#a wavy height function
z = np.sin(x) + np.cos(y)
points = np.column_stack((x, y, z))
get_best_fit_plane(points)

plt.close()




