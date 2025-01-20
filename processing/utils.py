import numpy as np
import pandas as pd
import torch

class SHREDDataset():
    """
    recon_data is a TimeSeriesDataset object for SHRED Reconstructor
    forecast_data is a TimeSeriesDataset object for SHRED Forecaster
    """
    def __init__(self, recon_data = None, forecast_data = None, time = None, sensor_measurements = None):
        self.reconstructor = recon_data
        self.forecaster = forecast_data
        self.time = time
        self.sensor_measurements = None


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
    Returns a dict of train, validation, and test indices.
    Input:
    - n: Effective number of timesteps. This is the number of time steps SHRED will be able perform reconstruction on.
         Currently this is equal to total number of time steps minus lags
    - train_size: A float representing size of training data
    - val_size: A float representing size of validation data
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

    train_indices = indices[:num_train_indices]
    val_indices = indices[num_train_indices:num_train_indices + num_val_indices]
    test_indices = indices[num_train_indices + num_val_indices:]

    return {
        "train": train_indices,
        "validation": val_indices,
        "test": test_indices,
    }

def get_data(file_path):
    """
    Returns a single numpy array from the given file_path
    """
    if file_path.endswith('.npz'):
        return get_data_npz(file_path)
    else:
        raise ValueError(f"Unsupported file type: '{file_path}'. Only .npz files are supported.")


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
    
def get_sensor_measurements(full_state_data, random_sensors, stationary_sensors, mobile_sensors):
    """
    - full_state_data: a nd numpy array with time on the last axis
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
    where sensor_measurements is a 2D numpy array with time on axis 1
    and sensor_summary is a pandas dataframe
    """
    sensor_summary = []
    sensor_measurements = []

    # generate random sensor locations
    if random_sensors is not None:
        if isinstance(random_sensors, int):
            random_sensor_locations = generate_random_sensor_locations(full_state = full_state_data, num_sensors = random_sensors)
            for sensor_coordinate in random_sensor_locations:
                sensor_summary.append(['stationary (randomly selected)', sensor_coordinate])
                sensor_measurements.append(full_state_data[sensor_coordinate])
        else:
            raise ValueError(f"Invalid `random_sensor`.")

    # selected stationary sensors
    if stationary_sensors is not None:
        if isinstance(stationary_sensors, tuple):
            stationary_sensors = [stationary_sensors]
        if all(isinstance(sensor, tuple) for sensor in stationary_sensors):
            for sensor_coordinate in stationary_sensors:
                sensor_summary.append(['stationary (user selected)', sensor_coordinate])
                sensor_measurements.append(full_state_data[sensor_coordinate])
        else:
            raise ValueError(f"Invalid `stationary_sensors`.")
        
    if mobile_sensors is not None:
        if isinstance(mobile_sensors[0], tuple):
            mobile_sensors = [mobile_sensors]
        if isinstance(mobile_sensors[0], list):
            for mobile_sensor_coordinates in mobile_sensors:
                if len(mobile_sensor_coordinates) != full_state_data.shape[-1]:
                    raise ValueError(
                        f"Number of mobile sensor coordinates ({len(mobile_sensor_coordinates)}) "
                        f"must match the number of timesteps ({full_state_data.shape[-1]})."
                    )
                sensor_summary.append(['mobile', mobile_sensor_coordinates])
                sensor_measurements.append([
                    full_state_data[sensor_coordinate][timestep]
                    for timestep, sensor_coordinate in enumerate(mobile_sensor_coordinates)
                ])
        else:
            raise ValueError(f"Invalid `mobile_sensors`.")
    sensor_measurements = np.array(sensor_measurements)
    sensor_summary = pd.DataFrame(sensor_summary, columns=["sensor type", "location/trajectory"])
    return {
        "sensor_measurements": sensor_measurements,
        "sensor_summary": sensor_summary,
    }


def generate_random_sensor_locations(full_state, num_sensors):
    """
    Input: a nd numpy array where the last axis is time and the number random sensor
    locations to return.
    Return: a list of sensor locations (tuples)
    """
    spatial_shape = full_state.shape[:-1] # last dimension always number of timesteps
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
    - full_state_data: a nd numpy array with time on the last axis
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

def get_sensor_measurements(full_state_data, random_sensors, stationary_sensors, mobile_sensors):
    """
    - full_state_data: a nd numpy array with time on the last axis
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
    where sensor_measurements is a 2D numpy array with time on axis 1
    and sensor_summary is a pandas dataframe
    """
    sensor_summary = []
    sensor_measurements = []

    # generate random sensor locations
    if random_sensors is not None:
        if isinstance(random_sensors, int):
            random_sensor_locations = generate_random_sensor_locations(full_state = full_state_data, num_sensors = random_sensors)
            for sensor_coordinate in random_sensor_locations:
                sensor_summary.append(['stationary (randomly selected)', sensor_coordinate])
                sensor_measurements.append(full_state_data[sensor_coordinate])
        else:
            raise ValueError(f"Invalid `random_sensor`.")

    # selected stationary sensors
    if stationary_sensors is not None:
        if isinstance(stationary_sensors, tuple):
            stationary_sensors = [stationary_sensors]
        if all(isinstance(sensor, tuple) for sensor in stationary_sensors):
            for sensor_coordinate in stationary_sensors:
                sensor_summary.append(['stationary (user selected)', sensor_coordinate])
                sensor_measurements.append(full_state_data[sensor_coordinate])
        else:
            raise ValueError(f"Invalid `stationary_sensors`.")
        
    if mobile_sensors is not None:
        if isinstance(mobile_sensors[0], tuple):
            mobile_sensors = [mobile_sensors]
        if isinstance(mobile_sensors[0], list):
            for mobile_sensor_coordinates in mobile_sensors:
                if len(mobile_sensor_coordinates) != full_state_data.shape[-1]:
                    raise ValueError(
                        f"Number of mobile sensor coordinates ({len(mobile_sensor_coordinates)}) "
                        f"must match the number of timesteps ({full_state_data.shape[-1]})."
                    )
                sensor_summary.append(['mobile', mobile_sensor_coordinates])
                sensor_measurements.append([
                    full_state_data[sensor_coordinate][timestep]
                    for timestep, sensor_coordinate in enumerate(mobile_sensor_coordinates)
                ])
        else:
            raise ValueError(f"Invalid `mobile_sensors`.")
    sensor_measurements = np.array(sensor_measurements)
    sensor_summary = pd.DataFrame(sensor_summary, columns=["sensor type", "location/trajectory"])
    return {
        "sensor_measurements": sensor_measurements,
        "sensor_summary": sensor_summary,
    }


def generate_lagged_sequences_from_sensor_measurements(sensor_measurements, lags):
    """
    Generates lagged sequences from sensor_measurments.
    Expects self.transformed_sensor_measurements to be a 2D numpy array with time is axis 0.
    Returns 3D numpy array of lagged sequences with timesteps along axis 0, lags along axis 1, sensors along axis 2.
    """
    num_timesteps = sensor_measurements.shape[0]
    num_sensors = sensor_measurements.shape[1]
    if num_timesteps <= lags:
        raise ValueError("Number of timesteps must be greater than the number of lags.")

    lagged_sequences = np.empty((num_timesteps - lags, lags + 1, num_sensors))
    for i in range(lagged_sequences.shape[0]):
        lagged_sequences[i] = sensor_measurements[i:i+lags+1, :]
    return lagged_sequences