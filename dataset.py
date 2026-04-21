import numpy as np
import matplotlib.pyplot as plt

# Normal data (tight cluster around [1,1])
normal_data = np.random.normal(loc=1, scale=0.2, size=(50, 2))

# Anomalies (opposite direction)
anomalies = np.random.normal(loc=-1, scale=0.2, size=(10, 2))

# Combine for visualization
all_data = np.vstack((normal_data, anomalies))

# Plot
plt.scatter(normal_data[:, 0], normal_data[:, 1], label="Normal")
plt.scatter(anomalies[:, 0], anomalies[:, 1], label="Anomaly")
plt.legend()
plt.title("Initial Dataset")
plt.show()