import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV files
unb_file_path = "unb.csv"
bin_file_path = "bin.csv"

# Read the CSV files
unb_df = pd.read_csv(unb_file_path)
bin_df = pd.read_csv(bin_file_path)

# Rename columns
unb_df.columns = ["Displacement", "Force_Unb"]
bin_df.columns = ["Displacement", "Force_Bin"]

# Drop NaNs
unb_df = unb_df.dropna()
bin_df = bin_df.dropna()

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot force data
ax1.plot(unb_df["Displacement"], unb_df["Force_Unb"], label="Unbind Force", color='red')
ax1.plot(bin_df["Displacement"], bin_df["Force_Bin"], label="Binding Force", color='blue')
ax1.set_xlabel("Displacement")
ax1.set_ylabel("Force", color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend(loc="upper left")

# Create second y-axis for bending energy
ax2 = ax1.twinx()
ax2.set_ylabel("Bending Energy", color='red')
ax2.scatter([193.8306, 193.6621], [5.94028173, 6.8604476], color='red', label='Unbind/Binding Bending Energy', marker='o')
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend(loc="upper right")

# Create third y-axis for stretching energy
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.set_ylabel("Stretching Energy", color='blue')
ax3.scatter([193.8306, 193.6621], [-0.0893271, -0.164777], color='blue', label='Unbind/Binding Stretching Energy', marker='*')
ax3.tick_params(axis='y', labelcolor='blue')
ax3.legend(loc="lower right")

# Adjust limits for scaling
ax1.set_ylim(min(unb_df["Force_Unb"].min(), bin_df["Force_Bin"].min()), max(unb_df["Force_Unb"].max(), bin_df["Force_Bin"].max()))
ax2.set_ylim(5, 7)
ax3.set_ylim(-0.2, 0)

# Show plot
plt.title("Force and Energy Plot")
plt.show()