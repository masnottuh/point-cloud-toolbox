import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd
from scipy.optimize import minimize
from matplotlib.patches import Patch

# Function to fit an implicit quadratic surface to the points
def fit_implicit_quadratic_surface(points):
    # Construct the design matrix for the quadratic form
    A = np.column_stack((
        points[:, 0]**2,  # x^2
        points[:, 1]**2,  # y^2
        points[:, 2]**2,  # z^2
        points[:, 0]*points[:, 1],  # xy
        points[:, 0]*points[:, 2],  # xz
        points[:, 1]*points[:, 2],  # yz
        points[:, 0],  # x
        points[:, 1],  # y
        points[:, 2],  # z
        np.ones(points.shape[0])  # Constant term
    ))

    # Define the objective function (least squares)
    def objective_function(coefficients):
        return np.sum((A @ coefficients)**2)

    # Define the constraint ||b|| - 1 = 0
    def constraint(coefficients):
        return np.linalg.norm(coefficients) - 1

    # Initial guess
    initial_guess = np.ones(A.shape[1])

    # Define the constraint as a dictionary
    cons = ({'type': 'eq', 'fun': constraint})

    # Solve the constrained minimization problem
    result = minimize(objective_function, initial_guess, constraints=cons)

    # Return the coefficients from the minimization result
    return result.x

# Function to plot the implicit surface
def plot_implicit_surface(points, coefficients):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=1)

    # Create a grid for the plotting range
    x_range = np.linspace(points[:, 0].min(), points[:, 0].max(), 50)
    y_range = np.linspace(points[:, 1].min(), points[:, 1].max(), 50)
    X, Y = np.meshgrid(x_range, y_range)

    # Coefficients for the implicit quadratic equation
    A, B, C, D, E, F, G, H, I, J = coefficients

    # Compute Z values on the grid
    Z_pos = np.zeros_like(X)
    Z_neg = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Quadratic equation in the form Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
            # We solve for z
            x, y = X[i, j], Y[i, j]
            # Coefficients for the quadratic equation az^2 + bz + c = 0
            a = C
            b = E*x + F*y + I
            c = A*x**2 + B*y**2 + D*x*y + G*x + H*y + J
            # Discriminant
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                # Calculate both roots
                Z_pos[i, j] = (-b + np.sqrt(discriminant)) / (2*a)
                Z_neg[i, j] = (-b - np.sqrt(discriminant)) / (2*a)
            else:
                # Assign NaN if no real root exists
                Z_pos[i, j] = np.nan
                Z_neg[i, j] = np.nan

    # Plot the positive root surface
    ax.plot_surface(X, Y, Z_pos, color='r', alpha=0.5, rstride=1, cstride=1, edgecolor='none')
    # Plot the negative root surface
    ax.plot_surface(X, Y, Z_neg, color='g', alpha=0.5, rstride=1, cstride=1, edgecolor='none')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Implicit Quadratic Surface with Both Roots')

    # Creating custom legend for 3D surfaces
    legend_positive = Patch(facecolor='red', edgecolor='r', alpha=0.5, label='Positive root')
    legend_negative = Patch(facecolor='green', edgecolor='g', alpha=0.5, label='Negative root')
    ax.legend(handles=[legend_positive, legend_negative], loc='upper right')

    plt.show()

################################################################################################################
################################################################################################################
################################################################################################################

num_points = 100
min_val, max_val = -1, 1

# Generate some synthetic data points
x = np.random.uniform(min_val, max_val, num_points)
y = np.random.uniform(min_val, max_val, num_points)


# sphere 
radius = 1
center = [0.1, 1, 0.88]
phi = np.random.uniform(0, np.pi, num_points)
theta = np.random.uniform(0, 2 * np.pi, num_points)
x = center[0] + radius * np.sin(phi) * np.cos(theta)
y = center[1] + radius * np.sin(phi) * np.sin(theta)
z = center[2] + radius * np.cos(phi)
points = np.column_stack((x, y, z))
coefficients = fit_implicit_quadratic_surface(points)
plot_implicit_surface(points, coefficients)

# torus
r = 1
R = 10
theta = np.random.uniform(0, 2*np.pi, num_points)
phi = np.random.uniform(0, 2*np.pi, num_points)
x = (R + r * np.cos(theta)) * np.cos(phi)
y = (R + r * np.cos(theta)) * np.sin(phi)
z = r * np.sin(theta)
points = np.column_stack((x, y, z))
coefficients = fit_implicit_quadratic_surface(points)
plot_implicit_surface(points, coefficients)



x = np.random.uniform(min_val, max_val, num_points)
y = np.random.uniform(min_val, max_val, num_points)

# plane at z=0, tilted
z = 0.1*x + 0.1*y
points = np.column_stack((x, y, z))
coefficients = fit_implicit_quadratic_surface(points)
plot_implicit_surface(points, coefficients)


#tilted parabaloid
z = (0.1*x)**2 + (0.1*y)**2 + 0.1*x
points = np.column_stack((x, y, z))
coefficients = fit_implicit_quadratic_surface(points)
plot_implicit_surface(points, coefficients)


#tilted saddle
z = x**2 - y**2
points = np.column_stack((x, y, z))
coefficients = fit_implicit_quadratic_surface(points)
plot_implicit_surface(points, coefficients)


#tilted monkey saddle
z = x**3 - 3*x*y**2
points = np.column_stack((x, y, z))
coefficients = fit_implicit_quadratic_surface(points)
plot_implicit_surface(points, coefficients)


#a wavy height function
z = np.sin(x) + np.cos(y)
points = np.column_stack((x, y, z))
coefficients = fit_implicit_quadratic_surface(points)
plot_implicit_surface(points, coefficients)



