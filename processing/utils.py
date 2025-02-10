import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch


class SHREDDataset():
    """
    recon_data is a TimeSeriesDataset object for SHRED Reconstructor
    forecast_data is a TimeSeriesDataset object for SHRED Forecaster
    """
    def __init__(self, reconstructor_dataset = None, predictor_dataset = None, sensor_forecaster_dataset = None):
        if reconstructor_dataset is not None:
            self.reconstructor_dataset = reconstructor_dataset
        if predictor_dataset is not None:
            self.predictor_dataset = predictor_dataset
        if sensor_forecaster_dataset is not None:
            self.sensor_forecaster_dataset = sensor_forecaster_dataset


class TimeSeriesDataset(torch.utils.data.Dataset):
    '''Takes input sequence of sensor measurements with shape (batch size, lags, num_sensors)
    and corresponding measurments of high-dimensional state, return Torch dataset'''
    def __init__(self, X, Y, time = None, sensor_measurements = None):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len


def get_train_val_test_indices(n, train_size, val_size, test_size, method):
    """
    Returns a dict of train, val, and test indices.
    Input:
    - n: Effective number of timesteps. This is the number of time steps SHRED will be able perform reconstruction on.
         Currently this is equal to total number of time steps minus lags
    - train_size: A float representing size of training data
    - val_size: A float representing size of val data
    - test_size: A float representing size of test data
    - method: Either "random" (generate indices for reconstructor) or "sequential" (generate indices for forecaster)
    """
    # validate input
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(f"train_size, val_size, test_size must sum to 1. "
                        f"Got {train_size + val_size + test_size:.2f}.")

    # generate indices
    if method == "random":
        indices = np.random.permutation(n)
    elif method == "sequential":
        indices = np.arange(n)
    else:
        raise ValueError(f"Invalid method '{method}'. Use 'random' or 'sequential'.")

    num_train_indices = int(train_size * n)
    num_val_indices = int(val_size * n)

    if num_train_indices == 0 or num_val_indices == 0 or (n - num_train_indices - num_val_indices) == 0:
        raise ValueError('Legnth of train_indices, val_indices, and test_indices each be length > 0.')
    train_indices = indices[:num_train_indices]
    val_indices = indices[num_train_indices:num_train_indices + num_val_indices]
    test_indices = indices[num_train_indices + num_val_indices:]
    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def get_data(data):
    """
    Takes in a file path (.npy or .npz) or a numpy array.
    Returns a single numpy array.
    """
    if isinstance(data, str):  # If input is a file path
        if data.endswith('.npz'):
            return get_data_npz(data)
        elif data.endswith('.npy'):
            return get_data_npy(data)
        else:
            raise ValueError(f"Unsupported file format: {data}. Only .npy and .npz files are supported.")
    elif isinstance(data, np.ndarray):
        return data  # Already a NumPy array, return as is
    else:
        raise ValueError(f"Unsupported input type: {type(data)}. Only .npy/.npz file paths or numpy arrays are supported.")

def get_data_npz(file_path):
    """
    Returns a single numpy array given a .npz file_path
    """
    data = np.load(file_path)
    # Handle cases with no arrays
    if len(data.files) == 0:
        raise ValueError(f"The .npz file '{file_path}' is empty. It must contain exactly one array.")
    # If the .npz file contains a single array, return it
    if len(data.files) == 1:
        return data[data.files[0]]
    else:
        raise ValueError(f"The .npz file '{file_path}' contains multiple arrays: {data.files}. It must contain exactly one array.")

def get_data_npy(file_path):
    """
    Loads and returns a numpy array from a .npy file.
    """
    return np.load(file_path)

def generate_random_sensor_locations(full_state, num_sensors):
    """
    Input: a nd numpy array where the first axis is time, and a int for number of random sensor
    locations to return.
    Return: a list of sensor locations (tuples)
    """
    spatial_shape = full_state.shape[1:] # first dimension is number of timesteps, rest is spatial dimentions
    spatial_points = np.prod(spatial_shape)
    sensor_indices = np.random.choice(spatial_points, size = num_sensors, replace = False)
    sensor_locations = []
    for sensor_index in sensor_indices:
        sensor_location = []
        for dim in reversed(spatial_shape):
            sensor_location.append(sensor_index % dim)
            sensor_index //= dim
        sensor_location = tuple(reversed(sensor_location))
        sensor_locations.append(sensor_location)
    return sensor_locations

def generate_lagged_sequences(lags, full_state_data = None, random_sensors = None, stationary_sensors = None, mobile_sensors = None, sensor_measurements = None):
    """
    - full_state_data: a nd numpy array with time on the first axis
    - random_sensors: number of randomly placed stationary sensors (integer)
    - stationary_sensors: coordinates of stationary sensors. Each sensor coordinate is a tuple.
                        If multiple stationary sensors, put tuples in a list (tuple or list of tuples).
    - mobile_sensors: coordinates (tuple) of a mobile sensor in a list (length of list should match number of timesteps in dataset).
                        if multiple mobile_sensors, use a nested list. (list of tuples, or nested list of tuples)
    returns: a dict with keys 'lagged_sequences' and 'sensor_summary'
    - lagged_sequences: lagged sequences with timesteps along axis 0, lags along axis 1, sensors along axis 2.
    - sensor_summary: sensor_summary is a pandas dataframe with features ["sensor type", "location/trajectory"].
    """
    if sensor_measurements is None:
        if full_state_data is None or (random_sensors is None and stationary_sensors is None and mobile_sensors is None):
            raise ValueError("Provide either `sensor_measurements` or full_state_data` along with sensor-related data.")
        sensor_measurements_dict = get_sensor_measurements(full_state_data, random_sensors, stationary_sensors, mobile_sensors)
        sensor_measurements = sensor_measurements_dict["sensor_measurements"]
        sensor_summary = sensor_measurements_dict["sensor_summary"]
    else:
        sensor_summary = None
    lagged_sequences = generate_lagged_sequences_from_sensor_measurements(sensor_measurements, lags)
    return {
        "lagged_sequences": lagged_sequences,
        "sensor_summary": sensor_summary,
    }

def get_sensor_measurements(full_state_data, id, random_sensors, stationary_sensors, mobile_sensors, time):
    """
    - full_state_data: a nd numpy array with time on the first axis (axis 0)
    - random_sensors: number of randomly placed stationary sensors (integer)
    - stationary_sensors: coordinates of stationary sensors. Each sensor coordinate is a tuple.
                        If multiple stationary sensors, put tuples in a list (tuple or list of tuples).
    - mobile_sensors: coordinates (tuple) of a mobile sensor in a list (length of list should match number of timesteps in dataset).
                        if multiple mobile_sensors, use a nested list. (list of tuples, or nested list of tuples)
    Out:
    dict:
    {
        "sensor_measurements": sensor_measurements,
        "sensor_summary": sensor_summary,
    }
    where sensor_measurements is a 2D numpy array with time on axis 0
    and sensor_summary is a pandas dataframe
    """
    sensor_summary = []
    sensor_measurements = []
    sensor_id_index = 0
    # generate random sensor locations
    random_sensor_locations = random_sensors
    if random_sensors is not None:
        if isinstance(random_sensors, int):
            random_sensor_locations = generate_random_sensor_locations(full_state = full_state_data, num_sensors = random_sensors)
        elif isinstance(random_sensors, tuple):
            random_sensor_locations = [random_sensors]
        if all(isinstance(sensor, tuple) for sensor in random_sensor_locations):
            for sensor_coordinate in random_sensor_locations:
                sensor_summary.append([id, id + '-' + str(sensor_id_index), 'stationary (randomly selected)', sensor_coordinate])
                sensor_measurements.append(full_state_data[(slice(None),) + sensor_coordinate]) # slice for time axis (all timesteps)
                sensor_id_index+=1
        else:
            raise ValueError(f"Invalid `random_sensor`.")

    # selected stationary sensors
    if stationary_sensors is not None:
        if isinstance(stationary_sensors, tuple):
            stationary_sensors = [stationary_sensors]
        if all(isinstance(sensor, tuple) for sensor in stationary_sensors):
            for sensor_coordinate in stationary_sensors:
                sensor_summary.append([id, id + '-' + str(sensor_id_index), 'stationary (user selected)', sensor_coordinate])
                sensor_measurements.append(full_state_data[(slice(None),) + sensor_coordinate])
                sensor_id_index+=1
        else:
            raise ValueError(f"Invalid `stationary_sensors`.")
        
    if mobile_sensors is not None:
        if isinstance(mobile_sensors[0], tuple):
            mobile_sensors = [mobile_sensors]
        if isinstance(mobile_sensors[0], list):
            for mobile_sensor_coordinates in mobile_sensors:
                if len(mobile_sensor_coordinates) != full_state_data.shape[0]:
                    raise ValueError(
                        f"Number of mobile sensor coordinates ({len(mobile_sensor_coordinates)}) "
                        f"must match the number of timesteps ({full_state_data.shape[0]})."
                    )
                sensor_summary.append([id, id + '-' + str(sensor_id_index), 'mobile', mobile_sensor_coordinates])
                sensor_measurements.append([
                    full_state_data[timestep][sensor_coordinate]
                    for timestep, sensor_coordinate in enumerate(mobile_sensor_coordinates)
                ])
                sensor_id_index+=1
        else:
            raise ValueError(f"Invalid `mobile_sensors`.")
    # transpose sensor_measurements so time up on axis 0, number of sensors on axis 1
    sensor_summary = None if len(sensor_summary) == 0 else pd.DataFrame(sensor_summary, columns=["field id", "sensor id", "sensor type", "location/trajectory"])
    if len(sensor_measurements) == 0:
        sensor_measurements = None
    else:
        sensor_measurements = np.array(sensor_measurements).T
        sensor_measurements = pd.DataFrame(sensor_measurements, columns = sensor_summary['sensor id'].tolist())
        if time is not None:
            sensor_measurements.insert(0, 'time', time)
    return {
        "sensor_measurements": sensor_measurements,
        "sensor_summary": sensor_summary,
    }



# def get_parametric_sensor_measurements(full_state_data, id, random_sensors, stationary_sensors, mobile_sensors, time, param):
#     results = get_sensor_measurements(full_state_data, id, random_sensors, stationary_sensors, mobile_sensors, time)
#     param_df = pd.DataFrame(param, columns=[f"param {i}" for i in range(traj.shape[2])])
#     # results["sensor_measurements"]
#     df_combined = pd.concat([results["sensor_measurements"], param_df], axis=1)




def generate_lagged_sequences_from_sensor_measurements(sensor_measurements, lags):
    """
    Generates lagged sequences from sensor_measurments.
    Expects self.transformed_sensor_measurements to be a 2D numpy array with time is axis 0.
    Returns 3D numpy array of lagged sequences with timesteps along axis 0, lags along axis 1, sensors along axis 2.
    """
    num_timesteps = sensor_measurements.shape[0]
    num_sensors = sensor_measurements.shape[1]
    # concatenate zeros padding at beginning of sensor data along axis 0
    sensor_measurements = np.concatenate((np.zeros((lags, num_sensors)), sensor_measurements), axis = 0)
    lagged_sequences = np.empty((num_timesteps, lags + 1, num_sensors))
    for i in range(lagged_sequences.shape[0]):
        lagged_sequences[i] = sensor_measurements[i:i+lags+1, :]
    return lagged_sequences


def generate_forecast_lagged_sequences_from_sensor_measurements(sensor_measurements, lags):
    """
    Generates forecast lagged sequences from sensor_measurments.
    Expects self.transformed_sensor_measurements to be a 2D numpy array with time is axis 0.
    Returns 3D numpy array of lagged sequences with timesteps along axis 0, lags along axis 1, sensors along axis 2.
    """
    num_timesteps = sensor_measurements.shape[0]
    num_sensors = sensor_measurements.shape[1]
    # concatenate zeros padding at beginning of sensor data along axis 0
    sensor_measurements = np.concatenate((np.zeros((lags+1, num_sensors)), sensor_measurements), axis = 0)
    lagged_sequences = np.empty((num_timesteps+1, lags + 1, num_sensors))
    for i in range(lagged_sequences.shape[0]):
        lagged_sequences[i] = sensor_measurements[i:i+lags+1, :]
    return lagged_sequences



# def fit_sensors(train_indices, sensor_measurements):
#     """
#     Takes in train_indices, method ("random" or "sequential")
#     Expects self.sensor_measurements to be a 2D nunpy array with time on axis 0.
#     Scaling: fits either MinMaxScaler or Standard Scaler.
#     Stores fitted scalers as object attributes.
#     """
#     # scaling full-state data
#     scaler = MinMaxScaler()
#     return scaler.fit(sensor_measurements[train_indices])


def transform_sensor(sensor_scaler, sensor_measurements):
    """
    Expects self.sensor_measurements to be a 2D nunpy array with time on axis 0.
    self.transformed_sensor_data to scaled scnsor_data (optional) have time on axis 0 (transpose).
    """
    # Perform scaling if all scaler-related attributes exist
    return sensor_scaler.transform(sensor_measurements)

def flatten(data):
    """
    Takes in a nd array where the time is along the axis 0.
    Flattens the nd array into 2D array with time along axis 0.
    """
    # Reshape the data: keep time (axis 0) and flatten the remaining dimensions
    return data.reshape(data.shape[0], -1)

def unflatten(data, spatial_shape):
    """
    Takes in a flatten array where time is along axis 0 and the a tuple spatial shape.
    Reshapes the flattened array into nd array using the provided time_dim and spatial_dim.
    """
    original_shape = (data.shape[0],) + spatial_shape
    return data.reshape(original_shape)




def l2(datatrue, datapred):
    datatrue = datatrue.to(torch.float64)
    datapred = datapred.to(torch.float64)
    norm_true = torch.linalg.norm(datatrue)
    norm_diff = torch.linalg.norm(datapred - datatrue)
    if norm_true == 0:  # Avoid division by zero
        return torch.tensor(float('nan'))  # Handle edge case
    return norm_diff / norm_true