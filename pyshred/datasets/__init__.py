import os
import numpy as np

# Define dataset paths
DATA_DIR = os.path.dirname(__file__)

# Load datasets
sst_data = np.load(os.path.join(DATA_DIR, "sst_data.npy"))
# Define `__all__` for controlled imports
__all__ = ["sst_data"]
