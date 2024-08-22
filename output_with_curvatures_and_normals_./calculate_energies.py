import numpy as np

def load_data(file_path):
    # Assuming CSV format: x, y, z, area, gaussian_curvature, mean_curvature
    data = np.genfromtxt(file_path, delimiter=',', names=True, dtype=None)
    return data

def calculate_energies(data):
    bending_energy = np.sum(data['area'] * (data['mean_curvature'] ** 2))
    stretching_energy = np.sum(data['area'] * data['gaussian_curvature'])
    return bending_energy, stretching_energy

# Usage example
file_path = 'exported_data.csv'  # Path to your exported CSV file
data = load_data(file_path)
bending_energy, stretching_energy = calculate_energies(data)
print(f"Bending Energy: {bending_energy}")
print(f"Stretching Energy: {stretching_energy}")
