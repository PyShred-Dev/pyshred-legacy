import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.extmath import randomized_svd
import torch
from sklearn.decomposition import PCA

# main purpose is fit, fit_transform, transform, and inverse_transform, only take the things it would need to do these.

class SHREDPreprocessor:


    def __init__(self, lags=20, compression=True, scaling="minmax"):
        """
        Initialize SHREDProcessor

        Parameters:
        - lags: int, default=20
            Number of timesteps to include in each lagged sequence, with the current timestep counted as one.

        - compression: bool, int, or float, default=True
            Dimensionality reduction using rSVD:
            - `False`: No compression.
            - `True`: Compression with a default of 20 components.
            - `int`: Number of components to retain.

        - scaling: str, default="minmax"
            Scaling method to apply to the data ("minmax", "standard").
        """
        self._validate_init_params(lags, compression, scaling)

        self.lags = lags
        self.compression = compression
        self.scaling = scaling
        
        if compression is True:
            self.n_components = 20
        elif isinstance(compression, int):
            self.n_components = compression
        else:
            self.n_components = None
        
        # Fitted transformations
        self.lagged_sequence_scalers = None
        self.full_state_scalers = None
        self.scaler_before_svd = None
        self.left_singular_values = None # U
        self.singular_values = None # S
        self.sensor_summary = None

        




    def generate_lagged_sequences(self, sensor_locations_dict = None, full_state_dict = None, sensor_measurements = None):
        result = generate_lagged_sequences(self.lags, full_state_dict, sensor_locations_dict, sensor_measurements)
        lagged_sequences = result["lagged_sequences"]
        sensor_summary = result["sensor_summary"]
        self.sensor_summary = sensor_summary
        return lagged_sequences

    # lagged_sequences (x)
    # full_state_dict (y)
    def train_test_split(self, lagged_sequences, full_state_dict, train_size = 0.8, method = "random"):
        return train_test_split(full_state_dict, lagged_sequences, train_size, method)

    # 1. Compress y: Standard Scaler, then rSVD (Optional,instance attributes: "compression",
    #                                           "n_components", "explained_variance_ratio")
    #    Saves the fitted Standard scaler and rSVD components as self.scaler_before_svd and self.rsvd_components
    # 2. Scale X: fit scalers lagged sequences (Optional, instance attributes: "scaling")
    #    Saves the fitted scaler as self.scaler_X
    # 3. Scale y: fit scalers on compressed/uncompressed full-date data (Optional, instance attributes: "scaling")
    #    Saves the fitted scaler as self.scaler_y
    def fit(self, lagged_sequences = None, full_state_dict = None):
        # compression
        if self.compression and self.n_components is not None:
            self.scaler_before_svd = {}
            self.left_singular_values = {}
            self.singular_values = {}
            for key, full_state_data in full_state_dict.items():
                # standard scale data
                sc = StandardScaler()
                sc.fit(full_state_data)
                full_state_data_scaled = sc.transform(full_state_data)
                self.scaler_before_svd[key] = sc
                # rSVD
                U, S, Vt = randomized_svd(full_state_data_scaled, n_components=self.n_components, n_iter='auto')
                self.left_singular_values[key] = U
                self.singular_values[key] = S

        # scaling full-state data
        if self.scaling is not None:
            self.full_state_scalers = {}
            scaler_class = MinMaxScaler if self.scaling == 'minmax' else StandardScaler
            for key, full_state_data in full_state_dict.items():
                scaler = scaler_class()
                self.full_state_scalers[key] = scaler.fit(full_state_data)
        
        # scaling lagged sequences
        if self.scaling is not None:
            self.lagged_sequence_scalers = {}
            scaler_class = MinMaxScaler if self.scaling == 'minmax' else StandardScaler
            if self.sensor_summary is not None:
                lagged_sequences_same_dataset = []
                start = 0
                current_dataset = self.sensor_summary.iloc[0]['dataset']
                for end in range(1, lagged_sequences.shape[-1]):
                    if self.sensor_summary.iloc[end]['dataset'] != current_dataset:
                        scaler = scaler_class()
                        self.lagged_sequence_scalers[k]

                    
            # run scaling

        # self.lags = lags
        # self.compression = compression
        # self.n_components = n_components
        # self.explained_variance_ratio = explained_variance_ratio
        # self.scaling = scaling

    # 1. Compress y: Standard Scaler, then rSVD (Optional,instance attributes: "compression",
    #                                           "n_components", "explained_variance_ratio")
    #    Saves the fitted Standard scaler and rSVD components as self.scaler_before_svd and self.rsvd_components
    # 2. Scale X: fit scalers lagged sequences (Optional, instance attributes: "scaling")
    #    Saves the fitted scaler as self.scaler_X
    # 3. Scale y: fit scalers on compressed/uncompressed full-date data (Optional, instance attributes: "scaling")
    #    Saves the fitted scaler as self.scaler_y
    # Returns transformed X and/or transformed y

    # Returns transformed X and/or transformed y
    # y should STILL be a dict, in and out
    # SHRED will take the dict and flatten during shred.fit
    def transform(self, lagged_sequences = None, full_state_dict = None):
        pass


    def fit_transform(self, lagged_sequences = None, full_state_dict = None):
        self.fit(lagged_sequences, full_state_dict)
        return self.transform(lagged_sequences, full_state_dict)


    # Inverse transform y to its original scale and dimensionality
    # 1. If scaling was applied, reverse scaling (instance attributes: "scaling", "scaler_y")
    # 2. If rSVD was applied, uncompress (instance attributes: "compression", "rsvd_components")
    # 3. If rSVD was applied, reverse scaling (instance attributes: "compression", "scaler_before_svd")
    def inverse_transform(y):
        pass


    def _validate_init_params(self, lags, compression, scaling):
        if not isinstance(lags, int) or lags <= 0:
            raise ValueError(f"Invalid 'lags' value: {lags}. Must be a positive integer.")

        if isinstance(compression, bool):
            pass
        elif isinstance(compression, int):
            if compression <= 0:
                raise ValueError(f"Invalid 'compression' value: {compression}. If an integer, must be a positive number.")
        else:
            raise ValueError(f"Invalid 'compression' value: {compression}. Must be a boolean or integer.")


        if scaling not in {"minmax", "standard", None}:
            raise ValueError(f"Invalid 'scaling' value: {scaling}. Use 'minmax', or None.")

# raw sensor data workflow:
# X = generate_lagged_sequences(sensor_measurements = {raw_data})
# X_transformed = transform(X)


# function & instance method
# Generate lagged sequences from full-state data and corresponding sensor locations,
# or directly from sensor measurements
def generate_lagged_sequences(lags, full_state_dict = None, sensor_locations_dict = None, sensor_measurements = None):
    if sensor_measurements is None:
        if full_state_dict is None or sensor_locations_dict is None:
            raise ValueError("Provide either `sensor_measurements` or both `full_state_dict` and `sensor_locations_dict`.")
        sensor_measurements_dict = sensor_locations_dict_to_sensor_measurements(full_state_dict, sensor_locations_dict)
        sensor_measurements = sensor_measurements_dict["sensor_measurements"]
        sensor_summary = sensor_measurements_dict["sensor_summary"]
    else:
        sensor_summary = None
    lagged_sequences = generate_lagged_sequences_from_sensor_measurements(sensor_measurements, lags)
    return {
        "lagged_sequences": lagged_sequences,
        "sensor_summary": sensor_summary,
    }


def generate_random_sensor_locations(full_state, num_sensors):
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



def sensor_locations_dict_to_sensor_measurements(full_state_dict, sensor_locations_dict):
    sensor_summary = []
    sensor_measurements = []
    for key, data in sensor_locations_dict.items():
        # generate random sensor locations
        if isinstance(data, int):
            data = generate_random_sensor_locations(full_state = full_state_dict[key], num_sensors = data)
        if isinstance(data[0], tuple):
            for sensor_coordinate in data:
                sensor_summary.append([key, 'stationary', sensor_coordinate])
                sensor_measurements.append(full_state_dict[key][sensor_coordinate])
        elif isinstance(data[0], list):

            for mobile_sensor_coordinates in data:
                if len(mobile_sensor_coordinates) != full_state_dict[key].shape[-1]:
                    raise ValueError(
                        f"Number of mobile sensor coordinates ({len(mobile_sensor_coordinates)}) "
                        f"must match the number of timesteps ({full_state_dict[key].shape[-1]})."
                    )
                sensor_summary.append([key, 'mobile', mobile_sensor_coordinates])
                sensor_measurements.append([
                    full_state_dict[key][sensor_coordinate][timestep]
                    for timestep, sensor_coordinate in enumerate(mobile_sensor_coordinates)
                ])
                # sensor_measurements.append(np.array([full_state_dict[key][sensor_coordinate] for 
                #                                      sensor_coordinate in mobile_sensor_coordinates]))
        else:
            raise ValueError(f"Unsupported sensor type for key {key}.")
    sensor_measurements = np.array(sensor_measurements)
    sensor_summary = pd.DataFrame(sensor_summary, columns=["dataset", "type", "location/trajectory"])
    return {
        "sensor_measurements": sensor_measurements,
        "sensor_summary": sensor_summary,
    }
# def _index_to_sensor_coordinate(index, spatial_shape):
#     if index < 0 or index >= np.prod(spatial_shape):
#         raise IndexError("Index out of bounds for the given spatial shape.")
    
#     coord = []
#     for dim in reversed(spatial_shape):
#         coord.append(index % dim)
#         index //= dim
#     return tuple(reversed(coord))




# all_data_in = np.zeros((n - lags, lags, num_sensors))
# for i in range(len(all_data_in)):
#     all_data_in[i] = transformed_X[i:i+lags, sensor_locations]




    # # 1. Compress y: Standard Scaler, then rSVD (Optional,instance attributes: "compression", 
    # #                                           "n_components", "explained_variance_ratio")
    # #    Saves the fitted Standard scaler and rSVD components as self.scaler_before_svd and self.rsvd_components
    # # 2. Scale X: fit scalers lagged sequences (Optional, instance attributes: "scaling")
    # #    Saves the fitted scaler as self.scaler_X
    # # 3. Scale y: fit scalers on compressed/uncompressed full-date data (Optional, instance attributes: "scaling")
    # #    Saves the fitted scaler as self.scaler_y
    # def fit(self, X = None, y = None):
    #     self.lags = lags
    #     self.compression = compression
    #     self.n_components = n_components
    #     self.explained_variance_ratio = explained_variance_ratio
    #     self.scaling = scaling

    # # 1. Compress y: Standard Scaler, then rSVD (Optional,instance attributes: "compression", 
    # #                                           "n_components", "explained_variance_ratio")
    # #    Saves the fitted Standard scaler and rSVD components as self.scaler_before_svd and self.rsvd_components
    # # 2. Scale X: fit scalers lagged sequences (Optional, instance attributes: "scaling")
    # #    Saves the fitted scaler as self.scaler_X
    # # 3. Scale y: fit scalers on compressed/uncompressed full-date data (Optional, instance attributes: "scaling")
    # #    Saves the fitted scaler as self.scaler_y
    # # Returns transformed X and/or transformed y
    # def fit_transform(X = None, y = None):
    #     pass


    # # Returns transformed X and/or transformed y
    # def transform(X = None, y = None):
    #     pass

    
    # # Inverse transform y to its original scale and dimensionality
    # # 1. If scaling was applied, reverse scaling (instance attributes: "scaling", "scaler_y")
    # # 2. If rSVD was applied, uncompress (instance attributes: "compression", "rsvd_components")
    # # 3. If rSVD was applied, reverse scaling (instance attributes: "compression", "scaler_before_svd")
    # def inverse_transform(y):
    #     pass


# raw sensor data workflow:
# X = generate_lagged_sequences(sensor_measurements = {raw_data})
# X_transformed = transform(X)



def prep_and_split(data, sensors, time = None, train_size = 0.8, lags = 20, n_components = 20, split = 'random'):
    data, sensors, time = _validate_input(data, sensors, time)

    # Flatten each n-dimensional array to 2D, with timesteps as the columns
    flattened_data = {key: arr.reshape(-1, arr.shape[-1]) for key, arr in data.items()}
    train_indices, test_indices = _generate_indices(num_timesteps = time.shape[0] - lags, train_size=train_size, method=split)
    sensor_data, sensor_scalers, sensor_summary  = _process_sensor_data(data, flattened_data, sensors, time, train_indices)
    # perform rSVD if n_components is not None
    if n_components is not None:
        print("Compressing Data...")
        flattened_data = _compress_data(flattened_data, n_components)
        print("Done.")
    # Transform scaled_data to have timesteps as rows
    for key in flattened_data:
        flattened_data[key] = flattened_data[key].T
    # Scaler fit on data with timesteps as rows
    data_scalers = {}
    for key, arr in flattened_data.items():
        sc = MinMaxScaler()
        data_scalers[key] = sc.fit(arr[train_indices])
        flattened_data[key] = sc.transform(arr)

    sensor_data_combined = np.hstack(list(flattened_data.values())) # Stack the scaled data horizontally with timesteps as rows
    sensor_data_combined = np.hstack((sensor_data, sensor_data_combined)) # Stack scaled sensor data with scaled data, with timesteps as rows
    return _generate_datasets(sensor_data_combined, len(sensor_summary), lags, train_indices, test_indices)


def _validate_input(data, sensors, time):
    print('validate input')
    if isinstance(data, str):
        with open(data, 'rb') as file:
            loaded_pickle = pickle.load(file)
            data = loaded_pickle
    if isinstance(sensors, str):
        with open(sensors, 'rb') as file:
            loaded_pickle = pickle.load(file)
            sensors = loaded_pickle
    # Check if 'data' argument is a dictionary
    if not isinstance(data, dict):
        raise TypeError(f"'data' must be a dictionary, but got {type(data).__name__}.")
    # Check if 'sensors' argument is a dictionary
    if not isinstance(sensors, dict):
        raise TypeError(f"'sensors' must be a dictionary, but got {type(sensors).__name__}.")
    # Check to make sure 'data' dictionary does not include reserved key 'sensors' 
    if any("sensors" in key for key in data.keys()):
        raise ValueError("The key 'sensors' is reserved and cannot be used in the 'data' dictionary.")
    # Check all arrays have the same size in the last dimension (number of timesteps)
    if len({arr.shape[-1] for arr in data.values()}) != 1:
        raise ValueError("The last dimension (number of timesteps) of each array in 'data' must be the same.")
    # Time argument is not None.
    if time is not None:
        # Check if time argument is a 1D numpy array
        if not isinstance(time, np.ndarray) or time.ndim != 1:
            raise ValueError("'time' must be a 1-dimensional numpy array.")
        # Check if time array is equally spaced
        if not np.all(np.equal(np.diff(time), np.diff(time)[0])):
            raise ValueError("'time' must contain equally spaced elements.")
        if not np.all(np.diff(time) > 0):
            raise ValueError("'time' must be in strictly increasing order.")
        # Check if length of time array matches the last dimension of data array
        if any(arr.shape[-1] != time.shape[0] for arr in data.values()):
            raise ValueError("The length of 'time' must match the last dimension (number of timesteps) of arrays in 'data'.")
    # Time argument is None.
    else:
        # Set time argument to span from 0 to size of the last dimension (number of timesteps) - 1
        time = np.arange(0, next(iter(data.values())).shape[-1], 1)
    return data, sensors, time


def _generate_indices(num_timesteps, train_size, method="random"):
    if train_size <= 0 or train_size > 1:
        raise ValueError("'train_size' must be in the range (0,1].")
    test_size = 1 - train_size

    if method == "random":
        # Generate random indices
        all_indices = np.random.permutation(num_timesteps)
    elif method == "sequential":
        # Generate sequential indices
        all_indices = np.arange(num_timesteps)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'random' or 'sequential'.")

    # Calculate sizes for each split
    num_train = int(train_size * num_timesteps)

    # Split indices
    train_indices = all_indices[:int(train_size * num_timesteps)]
    test_indices = all_indices[num_train:]

    return train_indices, test_indices

def _process_sensor_data(data, flattened_data, sensors, time, train_indices):
    ########################################### SENSORS PROCESSING #################################################
    all_sensor_data = np.empty((time.shape[0], 0))
    sensor_summary_list = []
    sensor_scalers = {}

    for key, sensor in sensors.items():
        # spatial_shape = flattened_data[key].shape[:-1]
        spatial_shape = data[key].shape[:-1]

            # Process stationary sensors
        if isinstance(sensor, int) or isinstance(sensor[0], tuple):
            sensor_data = _process_stationary_sensor(
                key, sensor, spatial_shape, flattened_data, sensor_summary_list
            )
        # Process mobile sensors
        elif isinstance(sensor[0], list):
            sensor_data = _process_mobile_sensor(
                key, sensor, time, spatial_shape, flattened_data, sensor_summary_list
            )
        else:
            raise ValueError(f"Unsupported sensor type for key {key}.")
        sc = MinMaxScaler()
        sensor_scalers[key] = sc.fit(sensor_data[train_indices,:])
        sensor_data = sc.transform(sensor_data)
        all_sensor_data = np.hstack((all_sensor_data, sensor_data))
    sensor_summary_features = ["dataset", "type", "location/trajectory"]
    sensor_summary = pd.DataFrame(sensor_summary_list, columns=sensor_summary_features)
    return all_sensor_data, sensor_scalers, sensor_summary


def _process_stationary_sensor(key, sensor, spatial_shape, flattened_data, sensor_summary_list):
    """
    Process stationary sensors.
    """
    # If 'sensor' is an integer, randomly select that many unique sensor locations
    if isinstance(sensor, int):
        row_dim = np.prod(spatial_shape) # size of flattened spatial dimension
        sensor_indices = np.random.choice(row_dim, size=sensor, replace=False)
        for sensor_index in sensor_indices:
            coord = _index_to_coord(sensor_index, spatial_shape)
            sensor_summary_list.append([key, 'stationary', coord])
    # If 'sensor' is a list of tuples/coordinates
    elif isinstance(sensor[0], tuple):
        sensor_indices = [_coord_to_index(coord, spatial_shape) for coord in sensor]
        for coord in sensor:
            sensor_summary_list.append([key, 'stationary', coord])
    else:
        raise ValueError(f"Invalid stationary sensor format for key {key}.")
    sensor_data = flattened_data[key][sensor_indices, :].T  # Timesteps as rows
    if sensor_data.ndim == 1: # if 1D reshape to 2D
        sensor_data.reshape(-1,1) # timesteps as rows
    return sensor_data

def _process_mobile_sensor(key, sensor, time, spatial_shape, flattened_data, sensor_summary_list):
    """
    Process mobile sensors.
    """
    sensor_data = None
    for mobile_sensor_coords in sensor: # a single mobile sensor is a list of tuples/coordinates
        # Check if length of mobile sensor measurments match the number of timesteps
        if len(mobile_sensor_coords) != time.shape[0]:
            raise ValueError(f"The length of mobile sensor measurements must match timesteps ({time.shape[0]}).")
        sensor_indices = [_coord_to_index(coord, spatial_shape) for coord in mobile_sensor_coords]
        mobile_sensor_data = flattened_data[key][sensor_indices, np.arange(time.shape[0])].T
        mobile_sensor_data = mobile_sensor_data.reshape(-1, 1) # reshape from 1D to 2D array with timesteps as rows
        sensor_summary_list.append([key, 'mobile', mobile_sensor_coords])
        # Aggregate mobile sensor data
        if sensor_data is None:
            sensor_data = mobile_sensor_data
        else:
            sensor_data = np.hstack((sensor_data, mobile_sensor_data))
    return sensor_data


def _index_to_coord(index, spatial_shape):
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


def _coord_to_index(coord, spatial_shape):
    """
    Given a coordinate and a spatial shape of an array,
    returns the row index of the coordinate assuming the array is flattened.
    """
    if len(spatial_shape) != len(coord):
        raise ValueError(f"Coordinate dimensions {len(coord)} must match spatial dimensions {len(spatial_shape)}. "\
                         f"\ncoord: {coord}, \nspatial_shape: {spatial_shape}")
    index = 0
    multiplier = 1
    for i, dim in zip(reversed(coord), reversed(spatial_shape)):
        if i < 0 or i >= dim:
            raise IndexError("Index out of bounds for dimension.")
        index += i * multiplier
        multiplier *= dim
    return index


# dependent on internal state, need self
def _compress_data(flattened_data, n_components):
    """Compresses the data using Randomized SVD."""
    u = {}
    s = {}
    v = {}
    for key, arr in flattened_data.items():
        arr = arr - np.mean(arr, axis=0)  # Center the data
        _u, _s, _v = randomized_svd(arr, n_components=n_components, n_iter="auto")
        u[key] = _u
        s[key] = _s
        v[key] = _v
    return v

def _generate_datasets(load_X, num_sensors, lags, train_indices, test_indices):
    ### GENERATE RECONSTRUCTOR DATASETS ###
    n = load_X.shape[0] # n is number to timesteps
    sensor_column_indices = np.arange(num_sensors)
    ### Generate input sequences to a SHRED model
    all_data_in = np.zeros((n - lags, lags + 1, num_sensors)) # # (lags + 1) because look back at lags number of frames e.g. if lags = 2, look at frame n - 2, n - 1, and n.
    for i in range(len(all_data_in)):
        # +1 because sensor measurement of reconstructed frame is also used
        # e.g. recon frame 10 with lags 2 means input sequence with frames 8, 9, and 10.
        all_data_in[i] = load_X[i:i+lags+1, sensor_column_indices] 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ### Data in
    X_train = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    X_test = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)
    ### Data out
    Y_train = torch.tensor(load_X[train_indices + lags], dtype=torch.float32).to(device)
    Y_test = torch.tensor(load_X[test_indices + lags], dtype=torch.float32).to(device)
    return X_train, X_test, Y_train, Y_test



# Function
# Takes in sensor measurements as a 2D numpy array, where each row represents a sensor's data,
# and each column corresponds to a specific timestep.
def generate_lagged_sequences_from_sensor_measurements(sensor_measurements, lags):
    num_sensors = sensor_measurements.shape[0]
    num_timesteps = sensor_measurements.shape[1]

    if num_timesteps <= lags:
        raise ValueError("Number of timesteps must be greater than the number of lags.")

    lagged_sequences = np.empty((num_timesteps - lags, lags + 1, num_sensors))
    for i in range(lagged_sequences.shape[0]):
        lagged_sequences[i] = sensor_measurements[:, i:i+lags+1].T
    return lagged_sequences

# Split data into training and testing sets, where X contains lagged sequences,
# and y represents the full-state data.
def train_test_split(full_state_dict, lagged_sequences, train_size = 0.8, method = "random"):
    num_timesteps_minus_lags = lagged_sequences.shape[0]
    if train_size <= 0 or train_size >= 1:
        raise ValueError("`train_size` must be in the range (0.0, 1.0).")
    # generate random indices for train/test
    if method == "random":
        indices = np.random.permutation(num_timesteps_minus_lags)
    elif method == "sequential":
        indices = np.arange(num_timesteps_minus_lags)
    else:
        raise ValueError(f"Invalid method '{method}'. Use 'random' or 'sequential'.")
    num_train_indices = int(train_size * num_timesteps_minus_lags)
    train_indices = indices[:num_train_indices]
    test_indices = indices[num_train_indices:]
    # Inputs
    X_train = lagged_sequences[train_indices]
    X_test = lagged_sequences[test_indices]
    # Outputs
    y_train = {}
    y_test = {}
    for key, full_state_data in full_state_dict.items():
        y_train[key] = full_state_data[..., train_indices]
        y_test[key] = full_state_data[..., test_indices]
    return X_train, X_test, y_train, y_test
