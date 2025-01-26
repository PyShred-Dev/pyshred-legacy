import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

class TimeSeriesDataset(torch.utils.data.Dataset):
    '''Takes input sequence of sensor measurements with shape (batch size, lags, num_sensors)
    and corresponding measurments of high-dimensional state, return Torch dataset'''
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len

def coord_to_index(coord, spatial_shape):
    """
    Given a coordinate and a spatial shape of an array,
    returns the row index of the coordinate assuming the array is flattened.
    """
    if len(spatial_shape) != len(coord):
        raise ValueError("Coordinate dimensions must match spatial dimensions.")
    index = 0
    multiplier = 1
    for i, dim in zip(reversed(coord), reversed(spatial_shape)):
        if i < 0 or i >= dim:
            raise IndexError("Index out of bounds for dimension.")
        index += i * multiplier
        multiplier *= dim
    return index

def index_to_coord(index, spatial_shape):
    """
    Given a row index of a flattened spatial array and the spatial shape of the array,
    returns the coordinate of the index assuming the array is flattened.
    """
    if index < 0 or index >= np.prod(spatial_shape):
        raise IndexError("Index out of bounds for the given spatial shape.")
    
    coord = []
    for dim in reversed(spatial_shape):
        coord.append(index % dim)
        index //= dim
    return tuple(reversed(coord))

# Generate indicies for training and validation dataset for reconstructor
# train and validation indices combined span from 0 to num_timesteps - lags - 1.
def generate_train_val_indices_reconstructor(num_timesteps, lags, val_size):
    train_size = 1 - val_size
    train_indices = np.random.choice(num_timesteps - lags, size = int(train_size * (num_timesteps-lags)), replace = False)
    mask = np.ones(num_timesteps - lags)
    mask[train_indices] = 0
    valid_indices = np.arange(0, num_timesteps - lags)[np.where(mask!=0)[0]]
    return train_indices, valid_indices

# Generate indicies for training and validation dataset for forecaster
# train and validation indices combined span from 0 to num_timesteps - lags - 1.
def generate_train_val_indices_forecaster(num_timesteps, lags, val_size):
    train_size = 1 - val_size
    train_indices = np.arange(0, int(train_size * (num_timesteps-lags)))
    valid_indices = np.arange(int(train_size * (num_timesteps-lags)), num_timesteps-lags)
    return train_indices, valid_indices

def generate_train_val_dataset_reconstructor(load_X, train_indices, valid_indices, lags, num_sensors):
    n = load_X.shape[0]  # n is number to timesteps
    sensor_column_indices = np.arange(num_sensors)
    ### Generate input sequences to a SHRED model
    all_data_in = np.zeros((n - lags, lags + 1, num_sensors)) # # (lags + 1) because look back at lags number of frames e.g. if lags = 2, look at frame n - 2, n - 1, and n.
    for i in range(len(all_data_in)):
        # +1 because sensor measurement of reconstructed frame is also used
        # e.g. recon frame 10 with lags 2 means input sequence with frames 8, 9, and 10.
        all_data_in[i] = load_X[i:i+lags+1, sensor_column_indices] 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ### Data in
    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)

    ### Data out
    train_data_out = torch.tensor(load_X[train_indices + lags], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(load_X[valid_indices + lags], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)

    return train_dataset, valid_dataset


def generate_train_val_dataset_sensor_forecaster(load_X, train_indices, valid_indices, lags, num_sensors):
    valid_indices = valid_indices[:-1] # remove last index because forecast looks at one frame ahead (or else out of bounds error)
    n = load_X.shape[0] # n is number to timesteps
    load_X = load_X[:, :num_sensors]
    ### Generate input sequences to a SHRED model
    # all_data_in = np.zeros((n - lags, lags + 1, num_sensors))
    all_data_in = np.zeros((n - lags - 1, lags + 1, num_sensors)) # (lags + 1) because look back at lags number of frames, (n - lags - 1) because forecast is one frame ahead
    for i in range(len(all_data_in)):
        # +1 because sensor measurement of reconstructed frame is also used
        # e.g. recon frame 10 with lags 2 means input sequence with frames 8, 9, and 10.
        all_data_in[i] = load_X[i:i+lags+1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ### Data in
    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)

    ### Data out: +1 to have output be one step ahead of final sensor measurement
    train_data_out = torch.tensor(load_X[train_indices + lags + 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(load_X[valid_indices + lags + 1], dtype=torch.float32).to(device)
    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)

    return train_dataset, valid_dataset



def unflatten(data, spatial_shape):
    """
    Reshape each row of data to the given spatial_shape and return 
    a numpy array with shape (spatial_shape, timesteps).

    Parameters:
    -----------
    data : np.ndarray
        A 2D numpy array where each row represents a timestep.
    spatial_shape : tuple
        A tuple representing the target shape for each row (timestep).

    Returns:
    --------
    reshaped_data : np.ndarray
        (*spatial_shape, timesteps).
    """
    # Number of timesteps (number of rows)
    timesteps = data.shape[0]
    
    # Reshape each row to the given spatial shape
    reshaped_data = np.array([row.reshape(spatial_shape) for row in data])
    # Move the timesteps dimension to the last position
    reshaped_data = np.moveaxis(reshaped_data, 0, -1)
    return reshaped_data