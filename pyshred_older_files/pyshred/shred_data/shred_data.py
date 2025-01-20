# import pickle
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.utils.extmath import randomized_svd
# import torch


# class TimeSeriesDataset(torch.utils.data.Dataset):
#     """
#     A PyTorch Dataset for time series data.
#     """
#     def __init__(self, data_in, data_out):
#         self.data_in = data_in
#         self.data_out = data_out

#     def __len__(self):
#         return len(self.data_in)

#     def __getitem__(self, index):
#         return self.data_in[index], self.data_out[index]
# """
#         Parameters:
#         -----------
#         data : str or dict, required
#             * A str representing a file path or directory path
#             * A dictionary of high-dimensional state space datasets where:
#             - The key(s) are strings used as dataset identifiers
#             - The values are numpy arrays representating the datasets.
#             The last dimension of each array must represent the number of timesteps.
        
#         sensors : dict, required
#             A dictionary of sensor locations where: 
#             - The key(s) are strings used as dataset identifiers
#             - The values can be one of the following:
#                 - **int**: The number of sensors. Sensors will be randomly placed.
#                 - **list of tuples**: For stationary sensors, where each tuple represents the
#                         location of a sensor in the spatial domain.
#                 - **list of lists of tuples**: For mobile sensors, where each list represents a
#                         sensors location over time. Each inner list contains tuples for each timestep,
#                         corresponding to the sensor's location at that timestep.

#         time : numpy array, optional
#             A 1D numpy array of evenly spaced, strictly increasing timestamps. Each element corresponds to a timestep
#             in the last dimension of the data arrays.
#             Default is a 1D numpy array ranging from 0 to `N-1`, where `N` is the size of the
#             last dimension of the dataset.
#         """
#             print('validate input')
#             if isinstance(self.data, str):
#                 with open(self.data, 'rb') as file:
#                     loaded_pickle = pickle.load(file)
#                     self.data = loaded_pickle
#             if isinstance(self.sensors, str):
#                 with open(self.sensors, 'rb') as file:
#                     loaded_pickle = pickle.load(file)
#                     self.sensors = loaded_pickle
#             # Check if 'data' argument is a dictionary
#             if not isinstance(self.data, dict):
#                 raise TypeError(f"'data' must be a dictionary, but got {type(self.data).__name__}.")
#             # Check if 'sensors' argument is a dictionary
#             if not isinstance(self.sensors, dict):
#                 raise TypeError(f"'sensors' must be a dictionary, but got {type(self.sensors).__name__}.")
#             # Check to make sure 'data' dictionary does not include reserved key 'sensors' 
#             if any("sensors" in key for key in self.data.keys()):
#                 raise ValueError("The key 'sensors' is reserved and cannot be used in the 'data' dictionary.")
#             # Check all arrays have the same size in the last dimension (number of timesteps)
#             if len({arr.shape[-1] for arr in self.data.values()}) != 1:
#                 raise ValueError("The last dimension (number of timesteps) of each array in 'data' must be the same.")
#             # Time argument is not None.
#             if self.time is not None:
#                 # Check if time argument is a 1D numpy array
#                 if not isinstance(self.time, np.ndarray) or self.time.ndim != 1:
#                     raise ValueError("'time' must be a 1-dimensional numpy array.")
#                 # Check if time array is equally spaced
#                 if not np.all(np.equal(np.diff(self.time), np.diff(self.time)[0])):
#                     raise ValueError("'time' must contain equally spaced elements.")
#                 if not np.all(np.diff(self.time) > 0):
#                     raise ValueError("'time' must be in strictly increasing order.")
#                 # Check if length of time array matches the last dimension of data array
#                 if any(arr.shape[-1] != self.time.shape[0] for arr in self.data.values()):
#                     raise ValueError("The length of 'time' must match the last dimension (number of timesteps) of arrays in 'data'.")
#             # Time argument is None.
#             else:
#                 # Set time argument to span from 0 to size of the last dimension (number of timesteps) - 1
#                 self.time = np.arange(0, next(iter(self.data.values())).shape[-1], 1)



# class ShredData:
#     def __init__(self, data, sensors = 3, time = None):
#         self.data = self.validate_data(data)
#         self.sensors = self.validate_sensors(sensors)
#         self.time = self.validate_time(time)
    
#     def validate_data(self, data):
#         if len(data) == 0:
#             raise ValueError("'data' dictionary is empty. Must contain at least one entry.")
#         if not isinstance(data, dict):
#             raise TypeError(f"'data' must be a dictionary, but got {type(data).__name__}.")
#         for key in data.keys():
#             if not isinstance(key, str):
#                 raise TypeError(f"All keys in 'data' must be strings, found {type(key).__name__} for key '{key}'.")
#         if any("sensors" in key for key in self.data.keys()):
#                 raise ValueError("The key 'sensors' is reserved and cannot be used in the 'data' dictionary.")
#         for key, arr in data.items():
#             if not isinstance(arr, np.ndarray):
#                 raise TypeError(f"The value for key '{key}' must be a numpy array, got {type(arr).__name__}.")
#         if len({arr.shape[-1] for arr in self.data.values()}) != 1:
#                 raise ValueError("The last dimension (number of timesteps) of each array in 'data' must be the same.")
        

#     def validate_sensors(self, sensors):
#     def validate_time(self, time):
        
#         # self.data = data
#         # self.sensors = sensors
#         # self.time = time
#         # self.recon_datasets = None
#         # self.forecast_datasets = None
#         # self.sensor_summary = None
#         # self.u = {}
#         # self.s = {}
#         # self.v = {}
#         # self.recon_train_dataset = None
#         # self.recon_val_dataset = None
#         # self.recon_test_dataset = None
#         # self.forecast_train_dataset = None
#         # self.forecast_val_dataset = None
#         # self.forecast_test_dataset = None
#         # self._sc_sensors = {}
#         # self._sc_data = {}

#         # # possibly sc (scalers)
#         # self._validate_input()
    
#     def prepare_datasets(self, val_size = 0.2, test_size = 0.2, lags = 20, n_components = 20):
#         self.lags = lags
#         self.n_components = n_components
#         # Flatten each n-dimensional array to 2D, with timesteps as the columns
#         flattened_data = {key: arr.reshape(-1, arr.shape[-1]) for key, arr in self.data.items()}
#         recon_train_indices, recon_val_indices, recon_test_indices = self._generate_indices(val_size, test_size, "random")
#         forecast_train_indices, forecast_val_indices, forecast_test_indices = self._generate_indices(val_size, test_size, "sequential")
#         sensor_data = self._process_sensor_data(flattened_data, recon_train_indices)
#         # perform rSVD if n_components is not None
#         if n_components is not None:
#             print("Compressing Data...")
#             flattened_data = self._compress_data(flattened_data)
#             print("Done.")
#         # Transform scaled_data to have timesteps as rows
#         for key in flattened_data:
#             flattened_data[key] = flattened_data[key].T
#         # Scaler fit on data with timesteps as rows
#         for key, arr in flattened_data.items():
#             sc = MinMaxScaler()
#             self._sc_data[key] = sc.fit(arr[recon_train_indices])
#             flattened_data[key] = sc.transform(arr)

#         sensor_data_combined = np.hstack(list(flattened_data.values())) # Stack the scaled data horizontally with timesteps as rows
#         sensor_data_combined = np.hstack((sensor_data, sensor_data_combined)) # Stack scaled sensor data with scaled data, with timesteps as rows
        
#         self._generate_recon_datasets(sensor_data_combined, recon_train_indices, recon_val_indices, recon_test_indices)
#         self._generate_forecast_datasets(sensor_data_combined, forecast_train_indices, forecast_val_indices, forecast_test_indices)
#         return self.recon_train_dataset, self.recon_val_dataset, self.recon_test_dataset, self.forecast_train_dataset, self.forecast_val_dataset, self.forecast_test_dataset




#     # def prepare_forecaster_datasets(self, val_size = 0.2, test_size = 0.2):


#     # def prepare_reconstructor_datasets(self, val_size = 0.2, test_size = 0.2, lags = 20, n_components = 20):
#     #     self.lags = lags
#     #     self.n_components = n_components
#     #     # Flatten each n-dimensional array to 2D, with timesteps as the columns
#     #     flattened_data = {key: arr.reshape(-1, arr.shape[-1]) for key, arr in self.data.items()}
#     #     recon_train_indices, recon_val_indices, recon_test_indices = self._generate_indices(val_size, test_size, "random")
#     #     forecast_train_indices, forecast_val_indices, forecast_test_indices = self._generate_indices(val_size, test_size, "sequential")
#     #     sensor_data = self._process_sensor_data(flattened_data)
#     #     # perform rSVD if n_components is not None
#     #     if n_components is not None:
#     #         print("Compressing Data...")
#     #         flattened_data = self._compress_data(flattened_data)
#     #         print("Done.")
    
#     #     for key in flattened_data:
#     #         flattened_data[key] = flattened_data[key].T
    
#     #     # loadX = self._process_sensor_data()

#     #     recon_train_data_in, recon_val_data_in, recon_test_data_in, recon_train_data_out, recon_val_data_out, recon_test_data_out = self._generate_recon_datasets(loadX, recon_train_indices, recon_val_indices, recon_test_indices)
#     #     forecast_train_data_in, forecast_val_data_in, forecast_test_data_in, forecast_train_data_out, forecast_val_data_out, forecast_test_data_out = self._generate_forecast_datasets(loadX, forecast_train_indices, forecast_val_indices, forecast_test_indices)

#     #     # self.train_datasets, self.val_datasets, self.test_datasets = self._generate_datasets(loadX, recon_train_indices, recon_val_indices, recon_test_indices, forecast_train_indices, forecast_val_indices, forecast_test_indices)
#     #     # return self.train_datasets, self.val_datasets, self.test_datasets


#     def _scale_reconstructor_data(self, recon_train_indices):
#         pass


#     def _scale_forecaster_data(self, forecast_train_indices):
#         pass


#     def _compress_data(self, flattened_data):
#         """Compresses the data using Randomized SVD."""
#         for key, arr in flattened_data.items():
#             arr = arr - np.mean(arr, axis=0)  # Center the data
#             u, s, v = randomized_svd(arr, n_components=self.n_components, n_iter="auto")
#             self.u[key] = u
#             self.s[key] = s
#             self.v[key] = v
#         return self.v


#     def _process_sensor_data(self, flattened_data, recon_train_indices):
#         ########################################### SENSORS PROCESSING #################################################
#         all_sensor_data = np.empty((self.time.shape[0], 0))
#         sensor_summary_list = []

#         for key, sensor in self.sensors.items():
#             spatial_shape = self.data[key].shape[:-1]

#              # Process stationary sensors
#             if isinstance(sensor, int) or isinstance(sensor[0], tuple):
#                 sensor_data = self._process_stationary_sensor(
#                     key, sensor, spatial_shape, flattened_data, sensor_summary_list
#                 )
#             # Process mobile sensors
#             elif isinstance(sensor[0], list):
#                 sensor_data = self._process_mobile_sensor(
#                     key, sensor, spatial_shape, flattened_data, sensor_summary_list
#                 )
#             else:
#                 raise ValueError(f"Unsupported sensor type for key {key}.")
#             sc = MinMaxScaler()
#             self._sc_sensors[key] = sc.fit(sensor_data[recon_train_indices,:])
#             sensor_data = sc.transform(sensor_data)
#             all_sensor_data = np.hstack((all_sensor_data, sensor_data))
#         sensor_summary_features = ["dataset", "type", "location/trajectory"]
#         self.sensor_summary = pd.DataFrame(sensor_summary_list, columns=sensor_summary_features)
#         return all_sensor_data


#         #     sc = MinMaxScaler()
#         #     self._sc_sensors[key] = sc.fit(sensor_data[train_indices_reconstructor,:])
#         #     sensor_data = sc.transform(sensor_data)
#         #     if all_sensor_data is None:
#         #         all_sensor_data = sensor_data
#         #     else:
#         #         all_sensor_data = np.hstack((all_sensor_data, sensor_data))

    
#         # sensor_summary_col_names = ["row index", "dataset", "type", "location/trajectory"]
#         # self.sensor_summary = pd.DataFrame(sensor_summary, columns=sensor_summary_col_names) # sensor location summary
#         # self.sensor_data = self._unscale_sensor_data(all_sensor_data.T)
#         # num_sensors = all_sensor_data.shape[1]

#         # ############################################## COMPRESSION #################################################
#         # # If 'compressed' is True, compress data using Randomized SVD.
#         # if self._compressed:
#         #     print("Compressing Data...")

#         #     for key, arr in data.items():
#         #         arr = arr - np.mean(arr, axis=0)  # Center the data (mean-normalize along the columns)
#         #         u, s, v = randomized_svd(arr, n_components=n_components, n_iter='auto')
#         #         self._u_dict[key] = u
#         #         self._s_dict[key] = s
#         #         self._v_dict[key] = v
#         # print("Done.")
        
#         # ############################################ FIT SCALERS AND TRANSFORM INPUT #####################################
#         # scaled_data = self._v_dict if self._compressed else data
#         # # Transform scaled_data to have timesteps as rows
#         # for key in scaled_data:
#         #     scaled_data[key] = scaled_data[key].T
#         # # Scaler fit on data with timesteps as rows
#         # for key, arr in scaled_data.items():
#         #     sc = MinMaxScaler()
#         #     self._sc_data[key] = sc.fit(arr[train_indices_reconstructor])
#         #     scaled_data[key] = sc.transform(arr)
        
#         # ############################################ DATA FOR SHRED #######################################################
#         # load_X = np.hstack(list(scaled_data.values())) # Stack the scaled data horizontally with timesteps as rows
#         # load_X = np.hstack((all_sensor_data, load_X)) # Stack scaled sensor data with scaled data, with timesteps as rows
#         # output_size = load_X.shape[1]
        
    
#     def _process_stationary_sensor(self, key, sensor, spatial_shape, flattened_data, sensor_summary_list):
#         """
#         Process stationary sensors.
#         """
#         # If 'sensor' is an integer, randomly select that many unique sensor locations
#         if isinstance(sensor, int):
#             row_dim = np.prod(spatial_shape) # size of flattened spatial dimension
#             sensor_indices = np.random.choice(row_dim, size=sensor, replace=False)
#             for sensor_index in sensor_indices:
#                 coord = index_to_coord(sensor_index, spatial_shape)
#                 sensor_summary_list.append([key, 'stationary', coord])
#         # If 'sensor' is a list of tuples/coordinates
#         elif isinstance(sensor[0], tuple):
#             sensor_indices = [coord_to_index(coord, spatial_shape) for coord in sensor]
#             for coord in sensor:
#                 sensor_summary_list.append([key, 'stationary', coord])
#         else:
#             raise ValueError(f"Invalid stationary sensor format for key {key}.")
#         sensor_data = flattened_data[key][sensor_indices, :].T  # Timesteps as rows
#         if sensor_data.ndim == 1: # if 1D reshape to 2D
#             sensor_data.reshape(-1,1) # timesteps as rows
#         return sensor_data


#     def _process_mobile_sensor(self, key, sensor, spatial_shape, flattened_data, sensor_summary_list):
#         """
#         Process mobile sensors.
#         """
#         sensor_data = None
#         for mobile_sensor_coords in sensor: # a single mobile sensor is a list of tuples/coordinates
#             # Check if length of mobile sensor measurments match the number of timesteps
#             if len(mobile_sensor_coords) != self.time.shape[0]:
#                 raise ValueError(f"The length of mobile sensor measurements must match timesteps ({self.time.shape[0]}).")
#             sensor_indices = [coord_to_index(coord, spatial_shape) for coord in mobile_sensor_coords]
#             mobile_sensor_data = flattened_data[key][sensor_indices, np.arange(self.time.shape[0])].T
#             mobile_sensor_data = mobile_sensor_data.reshape(-1, 1) # reshape from 1D to 2D array with timesteps as rows
#             sensor_summary_list.append([key, 'mobile', mobile_sensor_coords])
#             # Aggregate mobile sensor data
#             if sensor_data is None:
#                 sensor_data = mobile_sensor_data
#             else:
#                 sensor_data = np.hstack((sensor_data, mobile_sensor_data))
#         return sensor_data

#     def _generate_indices(self, val_size, test_size, method="random"):
#         """
#         Generate train, validation, and test indices for a dataset.

#         Parameters:
#         -----------
#         val_size : float
#             Proportion of data reserved for validation (0 < val_size < 1).
#         test_size : float
#             Proportion of data reserved for testing (0 < test_size < 1).
#         method : str, optional
#             Splitting method to use. Either "random" or "sequential". Default is "random".

#         Returns:
#         --------
#         train_indices : numpy.ndarray
#             Array of training indices.
#         val_indices : numpy.ndarray
#             Array of validation indices.
#         test_indices : numpy.ndarray
#             Array of testing indices.
#         """
#         if val_size + test_size >= 1:
#             raise ValueError("'val_size' and 'test_size' must sum to less than 1.")
        
#         train_size = 1 - val_size - test_size
#         total_indices = self.time.shape[0] - self.lags

#         if method == "random":
#             # Generate random indices
#             all_indices = np.random.permutation(total_indices)
#         elif method == "sequential":
#             # Generate sequential indices
#             all_indices = np.arange(total_indices)
#         else:
#             raise ValueError(f"Unknown method '{method}'. Use 'random' or 'sequential'.")

#         # Calculate sizes for each split
#         num_train = int(train_size * total_indices)
#         num_val = int(val_size * total_indices)

#         # Split indices
#         train_indices = all_indices[:num_train]
#         val_indices = all_indices[num_train:num_train + num_val]
#         test_indices = all_indices[num_train + num_val:]

#         return train_indices, val_indices, test_indices


#     def _generate_recon_datasets(self, load_X,recon_train_indices, recon_val_indices, recon_test_indices):
#         ### GENERATE RECONSTRUCTOR DATASETS ###
#         n = load_X.shape[0] # n is number to timesteps
#         num_sensors = len(self.sensor_summary)
#         sensor_column_indices = np.arange(num_sensors)
#         ### Generate input sequences to a SHRED model
#         all_data_in = np.zeros((n - self.lags, self.lags + 1, num_sensors)) # # (lags + 1) because look back at lags number of frames e.g. if lags = 2, look at frame n - 2, n - 1, and n.
#         for i in range(len(all_data_in)):
#             # +1 because sensor measurement of reconstructed frame is also used
#             # e.g. recon frame 10 with lags 2 means input sequence with frames 8, 9, and 10.
#             all_data_in[i] = load_X[i:i+self.lags+1, sensor_column_indices] 
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         ### Data in
#         recon_train_data_in = torch.tensor(all_data_in[recon_train_indices], dtype=torch.float32).to(device)
#         recon_val_data_in = torch.tensor(all_data_in[recon_val_indices], dtype=torch.float32).to(device)
#         recon_test_data_in = torch.tensor(all_data_in[recon_test_indices], dtype=torch.float32).to(device)
#         ### Data out
#         recon_train_data_out = torch.tensor(load_X[recon_train_indices + self.lags], dtype=torch.float32).to(device)
#         recon_val_data_out = torch.tensor(load_X[recon_val_indices + self.lags], dtype=torch.float32).to(device)
#         recon_test_data_out = torch.tensor(load_X[recon_test_indices + self.lags], dtype=torch.float32).to(device)
#         ### TimeSeriesDataset Objects
#         self.recon_train_dataset = TimeSeriesDataset(recon_train_data_in, recon_train_data_out)
#         self.recon_val_dataset = TimeSeriesDataset(recon_val_data_in, recon_val_data_out)
#         self.recon_test_dataset = TimeSeriesDataset(recon_test_data_in, recon_test_data_out)


#     def _generate_forecast_datasets(self, load_X, forecast_train_indices, forecast_val_indices, forecast_test_indices):
#         ### GENERATE FORECASTER DATASETS ###
#         n = load_X.shape[0] # n is number to timesteps
#         num_sensors = len(self.sensor_summary)
#         forecast_test_indices = forecast_test_indices[:-1] # remove last index because forecast looks at one frame ahead (or else out of bounds error)
#         load_X = load_X[:, :num_sensors]
#         ### Generate input sequences to a SHRED model
#         all_data_in = np.zeros((n - self.lags - 1, self.lags + 1, num_sensors)) # (lags + 1) because look back at lags number of frames, (n - lags - 1) because forecast is one frame ahead
#         for i in range(len(all_data_in)):
#             # +1 because sensor measurement of reconstructed frame is also used
#             # e.g. recon frame 10 with lags 2 means input sequence with frames 8, 9, and 10.
#             all_data_in[i] = load_X[i:i+self.lags+1]
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         ### Data in
#         forecast_train_data_in = torch.tensor(all_data_in[forecast_train_indices], dtype=torch.float32).to(device)
#         forecast_val_data_in = torch.tensor(all_data_in[forecast_val_indices], dtype=torch.float32).to(device)
#         forecast_test_data_in = torch.tensor(all_data_in[forecast_test_indices], dtype=torch.float32).to(device)
#         ### Data out: +1 to have output be one step ahead of final sensor measurement
#         forecast_train_data_out = torch.tensor(load_X[forecast_train_indices + self.lags + 1], dtype=torch.float32).to(device)
#         forecast_val_data_out = torch.tensor(load_X[forecast_val_indices + self.lags + 1], dtype=torch.float32).to(device)
#         forecast_test_data_out = torch.tensor(load_X[forecast_test_indices + self.lags + 1], dtype=torch.float32).to(device)
#         ### TimeSeriesDataset Objects
#         self.forecast_train_dataset = TimeSeriesDataset(forecast_train_data_in, forecast_train_data_out)
#         self.forecast_val_dataset = TimeSeriesDataset(forecast_val_data_in, forecast_val_data_out)
#         self.forecast_test_dataset = TimeSeriesDataset(forecast_test_data_in, forecast_test_data_out)


#     # def _generate_datasets(self, load_X,recon_train_indices, recon_val_indices, recon_test_indices, forecast_train_indices, forecast_val_indices, forecast_test_indices):
#     #         ### VARIABLES
#     #         n = load_X.shape[0] # n is number to timesteps

#     #         ### GENERATE RECONSTRUCTOR DATASETS ###
#     #         sensor_column_indices = np.arange(num_sensors)
#     #         ### Generate input sequences to a SHRED model
#     #         all_data_in = np.zeros((n - self.lags, self.lags + 1, num_sensors)) # # (lags + 1) because look back at lags number of frames e.g. if lags = 2, look at frame n - 2, n - 1, and n.
#     #         for i in range(len(all_data_in)):
#     #             # +1 because sensor measurement of reconstructed frame is also used
#     #             # e.g. recon frame 10 with lags 2 means input sequence with frames 8, 9, and 10.
#     #             all_data_in[i] = load_X[i:i+self.lags+1, sensor_column_indices] 
#     #         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     #         ### Data in
#     #         recon_train_data_in = torch.tensor(all_data_in[recon_train_indices], dtype=torch.float32).to(device)
#     #         recon_val_data_in = torch.tensor(all_data_in[recon_val_indices], dtype=torch.float32).to(device)
#     #         recon_test_data_in = torch.tensor(all_data_in[recon_test_indices], dtype=torch.float32).to(device)
#     #         ### Data out
#     #         recon_train_data_out = torch.tensor(load_X[recon_train_indices + self.lags], dtype=torch.float32).to(device)
#     #         recon_val_data_out = torch.tensor(load_X[recon_val_indices + self.lags], dtype=torch.float32).to(device)
#     #         recon_test_data_out = torch.tensor(load_X[recon_test_indices + self.lags], dtype=torch.float32).to(device)

#     #         ### GENERATE FORECASTER DATASETS ###
#     #         forecast_test_indices = forecast_test_indices[:-1] # remove last index because forecast looks at one frame ahead (or else out of bounds error)
#     #         load_X = load_X[:, :num_sensors]
#     #         ### Generate input sequences to a SHRED model
#     #         all_data_in = np.zeros((n - self.lags - 1, self.lags + 1, num_sensors)) # (lags + 1) because look back at lags number of frames, (n - lags - 1) because forecast is one frame ahead
#     #         for i in range(len(all_data_in)):
#     #             # +1 because sensor measurement of reconstructed frame is also used
#     #             # e.g. recon frame 10 with lags 2 means input sequence with frames 8, 9, and 10.
#     #             all_data_in[i] = load_X[i:i+self.lags+1]
#     #         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     #         ### Data in
#     #         forecast_train_data_in = torch.tensor(all_data_in[forecast_train_indices], dtype=torch.float32).to(device)
#     #         forecast_val_data_in = torch.tensor(all_data_in[forecast_val_indices], dtype=torch.float32).to(device)
#     #         forecast_test_data_in = torch.tensor(all_data_in[forecast_test_indices], dtype=torch.float32).to(device)
#     #         ### Data out: +1 to have output be one step ahead of final sensor measurement
#     #         forecast_train_data_out = torch.tensor(load_X[forecast_train_indices + self.lags + 1], dtype=torch.float32).to(device)
#     #         forecast_val_data_out = torch.tensor(load_X[forecast_val_indices + self.lags + 1], dtype=torch.float32).to(device)
#     #         forecast_test_data_out = torch.tensor(load_X[forecast_test_indices + self.lags + 1], dtype=torch.float32).to(device)

#     #         ### Create TimeSeriesDataset Objects
#     #         train_dataset = TimeSeriesDataset(recon_train_data_in, recon_train_data_out, forecast_train_data_in, forecast_train_data_out)
#     #         val_dataset = TimeSeriesDataset(recon_val_data_in, recon_val_data_out, forecast_val_data_in, forecast_val_data_out)
#     #         test_dataset = TimeSeriesDataset(recon_test_data_in, recon_test_data_out, forecast_test_data_in, forecast_test_data_out)

#     #         return train_dataset, val_dataset, test_dataset





#     def _validate_input(self):
#             print('validate input')
#             if isinstance(self.data, str):
#                 with open(self.data, 'rb') as file:
#                     loaded_pickle = pickle.load(file)
#                     self.data = loaded_pickle
#             if isinstance(self.sensors, str):
#                 with open(self.sensors, 'rb') as file:
#                     loaded_pickle = pickle.load(file)
#                     self.sensors = loaded_pickle
#             # Check if 'data' argument is a dictionary
#             if not isinstance(self.data, dict):
#                 raise TypeError(f"'data' must be a dictionary, but got {type(self.data).__name__}.")
#             # Check if 'sensors' argument is a dictionary
#             if not isinstance(self.sensors, dict):
#                 raise TypeError(f"'sensors' must be a dictionary, but got {type(self.sensors).__name__}.")
#             # Check to make sure 'data' dictionary does not include reserved key 'sensors' 
#             if any("sensors" in key for key in self.data.keys()):
#                 raise ValueError("The key 'sensors' is reserved and cannot be used in the 'data' dictionary.")
#             # Check all arrays have the same size in the last dimension (number of timesteps)
#             if len({arr.shape[-1] for arr in self.data.values()}) != 1:
#                 raise ValueError("The last dimension (number of timesteps) of each array in 'data' must be the same.")
#             # Time argument is not None.
#             if self.time is not None:
#                 # Check if time argument is a 1D numpy array
#                 if not isinstance(self.time, np.ndarray) or self.time.ndim != 1:
#                     raise ValueError("'time' must be a 1-dimensional numpy array.")
#                 # Check if time array is equally spaced
#                 if not np.all(np.equal(np.diff(self.time), np.diff(self.time)[0])):
#                     raise ValueError("'time' must contain equally spaced elements.")
#                 if not np.all(np.diff(self.time) > 0):
#                     raise ValueError("'time' must be in strictly increasing order.")
#                 # Check if length of time array matches the last dimension of data array
#                 if any(arr.shape[-1] != self.time.shape[0] for arr in self.data.values()):
#                     raise ValueError("The length of 'time' must match the last dimension (number of timesteps) of arrays in 'data'.")
#             # Time argument is None.
#             else:
#                 # Set time argument to span from 0 to size of the last dimension (number of timesteps) - 1
#                 self.time = np.arange(0, next(iter(self.data.values())).shape[-1], 1)
            
#             # self._data_keys = list(data.keys()) # list of keys in 'data' dictionary
#             # self._data_spatial_shape = {key: arr.shape[:-1] for key, arr in data.items()} # store the spatial shape (all dimensions except the last) of each array in the 'data' dictionary
#             # if self.n_components is None:
#             #     self._compressed = False
#             # else:
#             #     self._compressed = True
#             # self._num_timesteps = self.time.shape[0]

    

#     # def __init__(self, data, sensors, time = None, lags=40, n_components=20, val_size=0.2, test_size = 0.2):
#     #     """
#     #     Initialize the DataProcessor.

#     #     Parameters:
#     #     -----------
#     #     data : dict
#     #         High-dimensional state space datasets.
#     #     sensors : dict
#     #         Sensor locations or configurations.
#     #     time : numpy array or None, optional
#     #         Time array corresponding to timesteps in the data. If None, defaults to an evenly spaced array.
#     #     lags : int, optional
#     #         Number of timesteps the sequence model looks back. Default is 40.
#     #     n_components : int/None, optional
#     #         Number of principal components for dimensionality reduction. Default is 20.
#     #     val_size : float, optional
#     #         Proportion of data reserved for validation. Default is 0.2.
#     #     test_size : float, optional
#     #         Proportion of data reserved for testing. Default is 0.2.
#     #     """
#     #     self.data = data
#     #     self.sensors = sensors
#     #     self.time = time
#     #     self.lags = lags
#     #     self.n_components = n_components
#     #     self.val_size = val_size
#     #     self.test_size = test_size

#     #     self._data_spatial_shape = None
#     #     self._data_keys = None
#     #     self._compressed = None
#     #     self._num_timesteps = None




    

    

#     # def preprocess(self):
#     #     """
#     #     Perform preprocessing on the data.
#     #     - Flatten to 2D.
#     #     - Scale and transform.
#     #     - Apply dimensionality reduction if needed.
#     #     """
#     #     # Flatten data to 2D
#     #     self.flattened_data = {key: arr.reshape(-1, arr.shape[-1]) for key, arr in self.data.items()}

#     #     # Perform scaling
#     #     self.scalers = {}
#     #     self.scaled_data = {}
#     #     for key, arr in self.flattened_data.items():
#     #         scaler = MinMaxScaler()
#     #         self.scalers[key] = scaler
#     #         self.scaled_data[key] = scaler.fit_transform(arr.T).T

#     #     # Apply dimensionality reduction if required
#     #     if self.n_components is not None:
#     #         self.reduced_data = {}
#     #         for key, arr in self.scaled_data.items():
#     #             arr_centered = arr - np.mean(arr, axis=0)
#     #             u, s, v = randomized_svd(arr_centered, n_components=self.n_components, n_iter='auto')
#     #             self.reduced_data[key] = (u, s, v)
#     #     else:
#     #         self.reduced_data = self.scaled_data

#     def process_sensors(self):
#         """
#         Process the sensor data and generate sensor summaries.
#         """
#         # Logic to handle stationary and mobile sensors
#         # Create summaries as needed
#         pass

#     def generate_datasets(self):
#         """
#         Generate train and validation datasets.
#         """
#         # Logic to split data into training and validation datasets
#         pass

#     def get_train_val_data(self):
#         """
#         Returns processed train and validation data.
#         """
#         self.preprocess()
#         self.process_sensors()
#         return self.generate_datasets()

# def index_to_coord(index, spatial_shape):
#     """
#     Given a row index of a flattened spatial array and the spatial shape of the array,
#     returns the coordinate of the index assuming the array is flattened.
#     """
#     if index < 0 or index >= np.prod(spatial_shape):
#         raise IndexError("Index out of bounds for the given spatial shape.")
    
#     coord = []
#     for dim in reversed(spatial_shape):
#         coord.append(index % dim)
#         index //= dim
#     return tuple(reversed(coord))


# def coord_to_index(coord, spatial_shape):
#     """
#     Given a coordinate and a spatial shape of an array,
#     returns the row index of the coordinate assuming the array is flattened.
#     """
#     if len(spatial_shape) != len(coord):
#         raise ValueError("Coordinate dimensions must match spatial dimensions.")
#     index = 0
#     multiplier = 1
#     for i, dim in zip(reversed(coord), reversed(spatial_shape)):
#         if i < 0 or i >= dim:
#             raise IndexError("Index out of bounds for dimension.")
#         index += i * multiplier
#         multiplier *= dim
#     return index