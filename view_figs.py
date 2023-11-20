import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider
import pickle
import os
import matplotlib.style as mplstyle
mplstyle.use('fast')

file_directory = "./output/sridge/"
# files = [os.path.join(file_directory, file) for file in os.listdir(file_directory) if file.startswith("Gaussian")]
files = [os.path.join(file_directory, file) for file in os.listdir(file_directory) if file.endswith(".pickle")]

for file in files:
    fig = pickle.load(open(file, 'rb'))
    plt.show()