import matplotlib.pyplot as plt
import pickle
import os

file_directory = "./"
# files = [os.path.join(file_directory, file) for file in os.listdir(file_directory) if file.startswith("Gaussian")]
files = [os.path.join(file_directory, file) for file in os.listdir(file_directory) if file.endswith(".pickle")]

for file in files:
    fix = pickle.load(open(file, 'rb'))
    plt.show()