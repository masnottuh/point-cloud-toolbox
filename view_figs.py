import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider
import pickle
import os

file_directory = "./"
# files = [os.path.join(file_directory, file) for file in os.listdir(file_directory) if file.startswith("Gaussian")]
files = [os.path.join(file_directory, file) for file in os.listdir(file_directory) if file.endswith("1.pickle")]

for file in files:
    fig = pickle.load(open(file, 'rb'))
    plt.show()