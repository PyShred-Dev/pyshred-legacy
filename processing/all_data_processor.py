from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.extmath import randomized_svd
from .utils import *

class ParametricSHREDDataProcessor:
    """
    SHREDDataProcessor manages a single dataset.
    SHREDDataProcessor objects are created by the `add` method in SHREDDataManager.
    Methods:
    fit_transform: scales and compresses `data`
    generate_dataset: generates 
    """
    METHODS = {"random", "sequential"}

    def __init__(self, data, random_sensors, stationary_sensors, mobile_sensors, lags, time, compression, scaling, parametric, params, id):
        #TODO: allow for file path as well as prepared numpy arrays.
        #TODO: allow params to be based in (will be similar to extra sensors, should stay constant between trajectories)
        """
        Inputs:
        - manager: reference to SHREDDataManager object that created and initialized this SHREDDataProcessor
        - data: numpy array of shape (optional n_trajectories, n_time, x1, x2, x3,...,xN) where x is spatial dimensions
        - random_sensors: number of randomly placed stationary sensors (integer).
        - stationary_sensors: coordinates of stationary sensors. Each sensor coordinate is a tuple.
                              If multiple stationary sensors, put tuples into a list (tuple or list of tuples).
        - mobile_sensors: list of coordinates (tuple) for a mobile sensor (length of list should match number of timesteps in `data`).
                          If multiple mobile_sensors, use a nested list (list of tuples, or nested list of tuples).
        - time: 1D numpy array of timestamps
        - lags: number of time steps to look back (integer).
        - compression: dimensionality reduction (boolean or integer).
        - scaling: scaling settings ('minmax', 'standard').
        - params: dataset of parameters of shape (nparams, ntimes), time on last axis just like with data. (string or numpy array)
        - id: unique identifier for the dataset (string).
        """
        self.data = data
        # self.data = get_data(data) # full-state data where the last axis is time
        self.n_components = compression # DAVID THIS IS FOR TESTING ONLY TODO: add processing/val
        self.full_state_data = None
        self.params = params
        self.parametric = parametric
        if self.parametric is False:
            # Add a new axis at the beginning for trajectory (n_trajectory = 1)
            self.data = np.expand_dims(self.data, axis=0)

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
        self.original_shape = self.data.shape
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.sensor_measurements = None # sensor measurements of shape (n_sensors, t), time on last axis
        self.sensor_summary = None # summary of sensor locations

    # TODO: Move self.full_state_data from __init__ to a method called add or something
    # adding the very first trajectory/data should be no different from adding a
    # second trajectory. What should stay constant is the sensor location, and
    # number of params if any. Sensor measurements if passed in raw will change,
    # and the actual params (optional) values will change.
    # compress/scaled each trajectory seperately or at the end? Do they need
    # different scalers/compression components or same?
    # How should different trajectories be stored?

    # def add_trajectory():
    #     """
    #     Add a new trajectory (e.g. multiple experiments with different parameters).
    #     Expects sensors and field to be the same.
    #     """





    # def add_parameters():
    #     """
    #     problem if add it here because need to flatten earlier no?
    #     Not a problem! Add parameter associated to each trajectory only. Won't
    #     need a extra dimension for parameters since using add functions only...
    #     """

    # will not work with forecastor, only for reconstructor
    # TODO: SHRED wrapper will know not to fit forecastor if not forecastor data.
    # shouldn't have method, should by default choose random trajectories, otherwise use select, sequential not necessary
    

    def generate_dataset(self, train_indices, val_indices, test_indices, method):
        """
        Sets train, val, and test SHREDDataset objects with generated dataset.
        """
        # saves indices as attributes for adding new trajectory
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

        X_train, X_val, X_test = None, None, None
        y_train, y_val, y_test = None, None, None

        # Processing regarding X

        # Parametric Case:
        if self.parametric:
            for i in train_indices:
                sensor_measurements_dict = get_sensor_measurements(self.data[i], self.random_sensors, self.stationary_sensors, self.mobile_sensors) # get sensor data
                if self.sensor_measurements is None:
                    self.sensor_measurements = sensor_measurements_dict['sensor_measurements']
                else:
                    self.sensor_measurements = np.vstack((self.sensor_measurements, sensor_measurements_dict['sensor_measurements']))
                self.sensor_summary = sensor_measurements_dict['sensor_summary']

                if self.sensor_summary is None:
                    self.sensor_summary = sensor_measurements_dict['sensor_summary']
                else:
                    self.sensor_summary = pd.concat([self.sensor_summary, sensor_measurements_dict['sensor_summary']], 
                                                    axis = 0).reset_index(drop=True)
            # check if sensor_measurements exist
            if self.sensor_measurements.size != 0:
                    # fit
                    # self.fit_sensors(self.data[i].shape[0], method) # use all time indices of train trajectories
                    self.fit_sensors(len(self.time), method) # use all time indices of train trajectories
                    # transform
                    self.transform_sensor(method)
                    # generate X data
                    X_train, X_val, X_test = self.generate_X(train_indices, val_indices, test_indices, method)

        # Non-parametric Case:
        if self.parametric is False:
            for i in range(self.data.shape[0]):
                sensor_measurements_dict = get_sensor_measurements(self.data[i], self.random_sensors, self.stationary_sensors, self.mobile_sensors) # get sensor data
                if self.sensor_measurements is None:
                    self.sensor_measurements = sensor_measurements_dict['sensor_measurements']
                else:
                    self.sensor_measurements = np.vstack((self.sensor_measurements, sensor_measurements_dict['sensor_measurements']))
                self.sensor_summary = sensor_measurements_dict['sensor_summary']

                if self.sensor_summary is None:
                    self.sensor_summary = sensor_measurements_dict['sensor_summary']
                else:
                    self.sensor_summary = pd.concat([self.sensor_summary, sensor_measurements_dict['sensor_summary']], 
                                                    axis = 0).reset_index(drop=True)
            # check if sensor_measurements exist
            if self.sensor_measurements.size != 0:
                    # fit
                    self.fit_sensors(train_indices, method)
                    # transform
                    self.transform_sensor(method)
                    # generate X data
                    X_train, X_val, X_test = self.generate_X(train_indices, val_indices, test_indices, method)
        
        # parametric case


        
        # Processing regarding Y
        # flattens full state data into into 2D array with time along axis 0.
        for i in range(self.data.shape[0]):
            if self.full_state_data is None:
                self.full_state_data = self.flatten(self.data[i])
            else:
                self.full_state_data = np.vstack((self.full_state_data, self.flatten(self.data[i])))

        # fit (fit and transform can be combine with a wrapper or just integrate the code together)
        self.fit(train_indices, method)
        # transform
        self.transform(method)
        # generate y data
        y_train, y_val, y_test = self.generate_y(train_indices, val_indices, test_indices, method)
        self.full_state_data = None
        print('done generating dataset')
        # full_state_data = self.unflatten(full_state_data)
        if X_train is not None and X_val is not None and X_test is not None: # aka make sure sensor data exists, does not work for setting train, val, and test to None/0 yet
            return {
                'train': (X_train, y_train),
                'val': (X_val, y_val),
                'test': (X_test, y_test)
            }
        else:
            return {
                'train': (None, y_train),
                'val': (None, y_val),
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
            raise ValueError(f"Inval method '{method}'. Choose from {self.METHODS}.")
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
        # Transpose compressed data, time on axis 0
        # transformed_sensor_data = self.sensor_measurements.T
        # Perform scaling if all scaler-related attributes exist
        if self.sensor_scaler.get(method) is not None:
            self.transformed_sensor_data[method] = self.sensor_scaler[method].transform(self.sensor_measurements)
    

    def fit_transform(self, train_indices, method):
        pass
    
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
            raise ValueError(f"Inval method '{method}'. Choose from {self.METHODS}.")
        # compression
        if self.n_components is not None:
            # standard scale data
            sc = StandardScaler()
            sc.fit(self.full_state_data[train_indices, :])
            self.scaler_before_svd[method] = sc
            full_state_data_std_scaled = sc.transform(self.full_state_data)
            # rSVD
            print('full_state_data_std_scaled.shape should be ntimes * ntraj * train_size',full_state_data_std_scaled.shape)
            U, S, V = randomized_svd(full_state_data_std_scaled[train_indices, :], n_components=self.n_components, n_iter='auto')

            # TODO fit compression
            self.right_singular_values[method] = V
            self.left_singular_values[method] = U
            self.singular_values[method] = S
            compressed_full_state_data = self.full_state_data @ V.transpose()
            print("compressed full_state_data:", compressed_full_state_data.shape)

        # transpose self.full_state_data so time on axis 0
        # self.full_state_data = self.full_state_data.T
        # transpose V_component so time on axis 0
        # V_component = V_component.T


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
            transformed_data = self.scaler_before_svd[method].transform(self.full_state_data)
            # TODO: transformed_data now shape (t, nstate) DAVID LOOK HERE
            # use matteo's

            transformed_data = transformed_data @ np.transpose(self.right_singular_values.get(method))

            # s_matrix = np.diag(self.singular_values[method]) # diagonal matrix of singular values
            # s_inv = np.linalg.inv(s_matrix) # compute inverse of singular values matrix
            # transformed_data = np.dot(s_inv, np.dot(self.left_singular_values.get(method).T, transformed_data)) # calculate V_T

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
        val = lagged_sensor_sequences[val_indices]
        test = lagged_sensor_sequences[test_indices]
        return train, val, test
    
    def generate_y(self, train_indices, val_indices, test_indices, method):
        """
        Generates the target data for SHRED.
        The target data are full-state data that is compresssed (optional) and scaled (optional),
        and flattens the state data.
        Output: 2D numpy array with timesteps along axis 0 and flattened state data along axis 1.
        """
        # train = self.transformed_data[method][train_indices + self.lags]
        # val = self.transformed_data[method][val_indices + self.lags]
        # test = self.transformed_data[method][test_indices + self.lags]
        train = self.transformed_data[method][train_indices]
        val = self.transformed_data[method][val_indices]
        test = self.transformed_data[method][test_indices]
        return train, val, test
    
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

