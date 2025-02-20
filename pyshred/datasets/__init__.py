import numpy as np
import gzip
import os

# SST Demo Data (1000 timesteps)
DATA_DIR = os.path.dirname(__file__)
sst_data_path = os.path.join(DATA_DIR, "demo_sst.npy.gz")
with gzip.open(sst_data_path, 'rb') as f:
    sst_data = np.load(f)
__all__ = ["sst_data"]
