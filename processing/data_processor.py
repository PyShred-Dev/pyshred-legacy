from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.extmath import randomized_svd
from .utils import *

class SHREDDataProcessor:
    """
    SHREDDataProcessor manages a single dataset.
    SHREDDataProcessor objects are created by the `add` method in SHREDDataManager.
    Methods:
    fit_transform: scales and compresses `data`
    generate_dataset: generates 
    """
    METHODS = {"random", "sequential"}

    def __init__(self, data, random_sensors, stationary_sensors, mobile_sensors, lags, time, compression, scaling, id):
        """
        Inputs:
        - data: file path to a .npz file. (string)
        - random_sensors: number of randomly placed stationary sensors (integer).
        - stationary_sensors: coordinates of stationary sensors. Each sensor coordinate is a tuple.
                              If multiple stationary sensors, put tuples into a list (tuple or list of tuples).
        - mobile_sensors: list of coordinates (tuple) for a mobile sensor (length of list should match number of timesteps in `data`).
                          If multiple mobile_sensors, use a nested list (list of tuples, or nested list of tuples).
        - time: 1D numpy array of timestamps
        - lags: number of time steps to look back (integer).
        - compression: dimensionality reduction (boolean or integer).
        - scaling: scaling settings ('minmax', 'standard').
        - id: unique identifier for the dataset (string).
        """
        if compression is not None:
            if compression == True:
                self.n_components = 20
            elif isinstance(compression, int):
                self.n_components = compression
            else:
                raise ValueError('Compression must be a bool or an int.')
        self.full_state_data = get_data(data) # full-state data where the first axis is time (axis 0)
        self.random_sensors = random_sensors # number of randomly placed sensors
        self.stationary_sensors = stationary_sensors # stationary sensor locations
        self.mobile_sensors = mobile_sensors # mobile sensor locations
        self.lags = lags # number of timesteps to look back
        self.time = time # numpy array of time stamps associated with `data`
        self.scaling = scaling
        self.compression = compression
        self.sensor_scaler = {}
        self.transformed_sensor_data = {}
        self.scaler_before_svd = {}
        self.left_singular_values = {}
        self.singular_values = {}
        self.right_singular_values = {}
        self.scaler = {} # stores scaler of full-state data
        self.transformed_data = {}
        self.original_shape = self.full_state_data.shape

    def generate_dataset(self, train_indices, val_indices, test_indices, method):
        """
        Sets train, validation, and test SHREDDataset objects with generated dataset.
        """
        X_train, X_valid, X_test = None, None, None
        y_train, y_valid, y_test = None, None, None
        # get sensor data
        sensor_measurements_dict = get_sensor_measurements(self.full_state_data, self.random_sensors, self.stationary_sensors, self.mobile_sensors) # get sensor data
        self.sensor_measurements = sensor_measurements_dict['sensor_measurements']
        self.sensor_summary = sensor_measurements_dict['sensor_summary']
        # check if sensor_measurements exist
        if self.sensor_measurements.size != 0:
            # fit
            self.fit_sensors(train_indices, method)
            # transform
            self.transform_sensor(method)
            # generate X data
            # if method == 'random':
            X_train, X_valid, X_test = self.generate_X(train_indices, val_indices, test_indices, method)
            # elif method == 'sequential':
            #     X_train, X_valid, X_test = self.generate_X_forecaster(train_indices, val_indices, test_indices, method)
        if method == 'random':
            # flattens full state data into into 2D array with time along axis 1.
            self.full_state_data = self.flatten(self.full_state_data)
            print('self.full_state_data flattened', self.full_state_data.shape)
            # fit (fit and transform can be combine with a wrapper or just integrate the code together)
            self.fit(train_indices, method)
            # transform
            self.transform(method)
            # generate y data
            y_train, y_valid, y_test = self.generate_y(train_indices, val_indices, test_indices, method)
            self.full_state_data = self.unflatten(self.full_state_data)
        # get forecaster data if sensor measurements exist
        elif method == 'sequential' and self.sensor_measurements.size != 0:
            y_train, y_valid, y_test = self.generate_y_forecaster(X_train, X_valid, X_test)
            # remove final timestep of X since no y (next sensor measurement) exists for the final timestep
            X_train = X_train[:-1, :, :]
            X_valid = X_valid[:-1, :, :]
            X_test = X_test[:-1, :, :]
        if X_train is not None and X_valid is not None and X_test is not None: # aka make sure sensor data exists, does not work for setting train, validation, and test to None/0 yet
            return {
                'train': (X_train, y_train),
                'validation': (X_valid, y_valid),
                'test': (X_test, y_test)
            }
        else:
            return {
                'train': (None, y_train),
                'validation': (None, y_valid),
                'test': (None, y_test)
            }


    def fit_sensors(self, train_indices, method):
        """
        Takes in train_indices, method ("random" or "sequential")
        Expects self.sensor_measurements to be a 2D nunpy array with time on axis 0.
        Scaling: fits either MinMaxScaler or Standard Scaler.
        Stores fitted scalers as object attributes.
        """
        if method not in self.METHODS:
            raise ValueError(f"Invalid method '{method}'. Choose from {self.METHODS}.")

        # scaling full-state data
        if self.scaling is not None:
            scaler_class = MinMaxScaler if self.scaling == 'minmax' else StandardScaler
            scaler = scaler_class()
            self.sensor_scaler[method] = scaler.fit(self.sensor_measurements[train_indices])


    def transform_sensor(self, method):
        """
        Expects self.sensor_measurements to be a 2D nunpy array with time on axis 0.
        self.transformed_sensor_data to scaled scnsor_data (optional) have time on axis 0 (transpose).
        """
        # Perform scaling if all scaler-related attributes exist
        if self.sensor_scaler.get(method) is not None:
            self.transformed_sensor_data[method] = self.sensor_scaler[method].transform(self.sensor_measurements)
    
    
    def fit(self, train_indices, method):
        """
        Input: train_indices and method ("random" or "sequential")
        Expects self.data to be flattened with time on axis 0.
        Compression: fits standard scaler and save left singular values and singular values
        Scaling: fits either MinMaxScaler or Standard Scaler.
        Stores fitted scalers as object attributes.
        """
        V_component = None
        if method not in self.METHODS:
            raise ValueError(f"Invalid method '{method}'. Choose from {self.METHODS}.")
        # compression
        if self.n_components is not None:
            # standard scale data
            sc = StandardScaler()
            sc.fit(self.full_state_data[train_indices, :])
            self.scaler_before_svd[method] = sc
            full_state_data_std_scaled = sc.transform(self.full_state_data)
            # rSVD
            U, S, V = randomized_svd(full_state_data_std_scaled[train_indices, :], n_components=self.n_components, n_iter='auto')
            print('n_components', self.n_components)
            print('V.shape', V.shape)
            self.right_singular_values[method] = V
            self.left_singular_values[method] = U
            self.singular_values[method] = S

            compressed_full_state_data = self.full_state_data @ V.transpose()

            print("compressed full_state_data:", compressed_full_state_data.shape)
        # transpose self.full_state_data so time on axis 0
        # self.full_state_data = self.full_state_data.T
        # transpose V_component so time on axis 0
        # V_component = V_component.T

        # try:
        #     # scaling full-state data
        #     if self.scaling is not None:
        #         scaler_class = MinMaxScaler if self.scaling == 'minmax' else StandardScaler
        #         scaler = scaler_class()
        #         if self.n_components is not None:
        #             self.scaler[method] = scaler.fit(V_component)
        #         else:
        #             self.scaler[method] = scaler.fit(self.full_state_data[train_indices])

        # # un-tranpose self.full_state_data so time on axis 1
        # finally:
        #     self.full_state_data = self.full_state_data.T

        
        # scaling full-state data

        if self.scaling is not None:
            scaler_class = MinMaxScaler if self.scaling == 'minmax' else StandardScaler
            scaler = scaler_class()
            if self.n_components is not None:
                self.scaler[method] = scaler.fit(compressed_full_state_data[train_indices])
            else:
                self.scaler[method] = scaler.fit(self.full_state_data[train_indices])

    def transform(self, method):
        """
        Expects self.full_state_data to be flattened with time on axis 0.
        Generates transfomed data which is self.full_state_data compressed (optional) and scaled (optional).
        """
        # Perform compression if all compression-related attributes exist
        if self.right_singular_values.get(method) is not None and self.scaler_before_svd[method] is not None:
            print('self.full_state_data pre transform', self.full_state_data.shape)
            transformed_data = self.scaler_before_svd[method].transform(self.full_state_data)
            print('transformed_data_std_scale', transformed_data.shape)
            transformed_data = transformed_data @ np.transpose(self.right_singular_values.get(method))
            print('transformed_data', transformed_data.shape)
            # s_matrix = np.diag(self.singular_values[method]) # diagonal matrix of singular values
            # s_inv = np.linalg.inv(s_matrix) # compute inverse of singular values matrix
            # transformed_data = np.dot(s_inv, np.dot(self.left_singular_values.get(method).T, transformed_data)) # calculate V_T
        else:
            transformed_data = self.full_state_data
        # Transpose compressed data, time on axis 0
        # transformed_data = transformed_data.T

        # Perform scaling if all scaler-related attributes exist
        if self.scaler.get(method) is not None:
            self.transformed_data[method] = self.scaler[method].transform(transformed_data)


    def generate_X(self, train_indices, val_indices, test_indices, method):
        """
        Generates the input data for SHRED.
        Expects self.sensor_measurements to be a 2D numpy array with time is axis 0.
        Output: 3D torch.tensor with timesteps along axis 0, lags along axis 1, sensors along axis 2.
        Output: 3D numpy arrays with timesteps along axis 0, lags along axis 1, sensors along axis 2.
        """
        # need to pass in the first lags number of data set well though
        lagged_sensor_sequences = generate_lagged_sequences_from_sensor_measurements(self.transformed_sensor_data[method], self.lags)
        train = lagged_sensor_sequences[train_indices]
        valid = lagged_sensor_sequences[val_indices]
        test = lagged_sensor_sequences[test_indices]
        return train, valid, test
    
    # def generate_X_forecaster(self, train_indices, val_indices, test_indices, method):
    #     """
    #     Generates the input data for SHRED.
    #     Expects self.sensor_measurements to be a 2D numpy array with time is axis 0.
    #     Output: 3D torch.tensor with timesteps along axis 0, lags along axis 1, sensors along axis 2.
    #     Output: 3D numpy arrays with timesteps along axis 0, lags along axis 1, sensors along axis 2.
    #     """
    #     # need to pass in the first lags number of data set well though
    #     # add one so forecaster can predict the very first frame
    #     # padding is expanded by one, so first prediction is the very first non-padded value
    #     lagged_sensor_sequences = generate_lagged_sequences_from_sensor_measurements(self.transformed_sensor_data[method], self.lags+1)
    #     train = lagged_sensor_sequences[train_indices+1]
    #     valid = lagged_sensor_sequences[val_indices+1]
    #     test = lagged_sensor_sequences[test_indices+1]
    #     return train, valid, test
    
    def generate_y_forecaster(self, X_train, X_valid, X_test): 
        train = X_train[1:, -1, :]  # Use the sensor measurements at the next timestep (1399, 5)
        valid = X_valid[1:, -1, :]
        test = X_test[1:, -1, :]
        return train, valid,test

    def generate_y(self, train_indices, val_indices, test_indices, method):
        """
        Generates the target data for SHRED.
        The target data are full-state data that is compresssed (optional) and scaled (optional),
        and flattens the state data.
        Output: 2D numpy array with timesteps along axis 0 and flattened state data along axis 1.
        """
        # train = self.transformed_data[method][train_indices + self.lags]
        # valid = self.transformed_data[method][val_indices + self.lags]
        # test = self.transformed_data[method][test_indices + self.lags]
        train = self.transformed_data[method][train_indices]
        valid = self.transformed_data[method][val_indices]
        test = self.transformed_data[method][test_indices]
        return train, valid, test

    def flatten(self, data):
        """
        Takes in a nd array where the time is along the axis 0.
        Flattens the nd array into 2D array with time along axis 0.
        """
        self.original_shape = data.shape
        # Reshape the data: keep time (axis 0) and flatten the remaining dimensions
        return data.reshape(data.shape[0], -1)
    
    def unflatten(self, data):
        """
        Takes in a flatten array where time is along axis 0 and the a tuple spatial shape.
        Reshapes the flattened array into nd array using the provided spatial shape,
        where time is along the last axis.
        """
        if self.original_shape is None:
            raise ValueError("Original shape not available.")
        return data.reshape(self.original_shape)
    
    def discard_data(self):
        self.full_state_data = None
        self.transformed_data = None
        self.transformed_sensor_data = None

