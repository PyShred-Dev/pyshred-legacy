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

    def __init__(self, data, random_sensors, stationary_sensors, mobile_sensors, lags, time, compression, scaling, params, id):
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
        if compression is not None:
            if compression == True:
                self.n_components = 20
            elif isinstance(compression, int):
                self.n_components = compression
            else:
                raise ValueError('Compression must be a bool or an int.')
        self.data = get_data(data)
        self.ntrajectories = self.data.shape[0]
        self.ntimes = len(time)
        if params is not None:
            self.nparams = params.shape[2]
        else:
            self.nparams = 0
        # self.data = get_data(data) # full-state data where the last axis is time
        self.n_components = compression # DAVID THIS IS FOR TESTING ONLY TODO: add processing/validation
        # self.full_state_data = None
        self.params = params

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
        # self.original_shape = self.data.shape

        self.sensor_measurements = None # sensor measurements of shape (n_sensors, t), time on last axis
        self.sensor_summary = None # summary of sensor locations
        self.sensor_summary = None
        self.sensor_measurements = None
        self.sensor_measurements_pd = None
        self.id = str(id)
        self.Y_spatial_dim = None
        self.random_sensor_locations = None

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
        Sets train, validation, and test SHREDDataset objects with generated dataset.
        """
        X_train, X_valid, X_test = None, None, None
        y_train, y_valid, y_test = None, None, None

        # Processing regarding X
        # Parametric Case:
        all_traj_sensor_measurements = None

        for i in range(self.data.shape[0]):
            single_traj_data = self.data[i]

                
            if i == 0:
                if self.random_sensors is not None:
                    if isinstance(self.random_sensors, int):
                        self.random_sensor_locations = generate_random_sensor_locations(full_state = single_traj_data, num_sensors = self.random_sensors)
            
                self.stationary_sensors = self.stationary_sensors + self.random_sensor_locations
            sensor_measurements_dict = get_sensor_measurements(full_state_data=single_traj_data,
                                                               id = self.id, time = self.time,
                                                               stationary_sensors=self.stationary_sensors,
                                                               mobile_sensors=self.mobile_sensors,
                                                               random_sensors=None) # get sensor data
            if sensor_measurements_dict['sensor_measurements'] is not None:
                sensor_measurements = sensor_measurements_dict['sensor_measurements'].drop(columns=['time']).to_numpy()[np.newaxis, :] # add trajectory axis
                sensor_measurements_pd = sensor_measurements_dict['sensor_measurements']
                sensor_measurements_pd.insert(0, "trajectory", i)
                if self.params is not None:
                    single_param_data = self.params[i]
                    single_param_data_df = pd.DataFrame(single_param_data, columns=[f"{self.id} param {i}" for i in range(single_param_data.shape[1])])
                    sensor_measurements_pd = pd.concat([sensor_measurements_pd, single_param_data_df], axis=1)
                sensor_summary = sensor_measurements_dict['sensor_summary']
    
                
                # sensor_measurements_dict['sensor_measurements'] = sensor_measurements_dict['sensor_measurements'].drop(columns=['time']).to_numpy()
                # sensor_measurements_dict['sensor_measurements'] = sensor_measurements_dict['sensor_measurements'][np.newaxis, :, :]  # add trajectory axis
                # self.sensor_summary = sensor_measurements_dict['sensor_summary']

                # Add a new axis to array2 to match dimensions
                # sensor_measurements_dict['sensor_measurements'] = sensor_measurements_dict['sensor_measurements'][np.newaxis, :, :]  # add trajectory axis
                # print("sensor_measurements_dict['sensor_measurements'].shape:",sensor_measurements_dict['sensor_measurements'].shape)
                if all_traj_sensor_measurements is None:
                    all_traj_sensor_measurements = sensor_measurements
                    self.nsensors = all_traj_sensor_measurements.shape[2]
                else:
                    all_traj_sensor_measurements = np.concatenate((all_traj_sensor_measurements, sensor_measurements), axis = 0)
                if self.sensor_summary is None:
                    self.sensor_summary = sensor_summary
                # print('sensor_measurements_pd', sensor_measurements_pd)
                if self.sensor_measurements_pd is None:
                    self.sensor_measurements_pd = sensor_measurements_pd
                else:
                    self.sensor_measurements_pd = pd.concat([self.sensor_measurements_pd, sensor_measurements_pd], axis=0, ignore_index=True)
                



        # print('all_traj_sensor_measurements', all_traj_sensor_measurements.shape)
        # print('sensor summary', self.sensor_summary)
            # else:
            #     self.sensor_summary = pd.concat([self.sensor_summary, sensor_measurements_dict['sensor_summary']], 
            #                                     axis = 0).reset_index(drop=True)
        # check if sensor_measurements exist
        if all_traj_sensor_measurements is not None:
            if self.params is not None:
                all_traj_sensor_measurements_and_params = np.concatenate((all_traj_sensor_measurements, self.params), axis=2)  # Shape (4, 2000, 3)
            else:
                all_traj_sensor_measurements_and_params = all_traj_sensor_measurements
            train_traj_sensor_measurements_and_params = all_traj_sensor_measurements_and_params[train_indices]
            valid_traj_sensor_measurements_and_params = all_traj_sensor_measurements_and_params[val_indices]
            test_traj_sensor_measurements_and_params = all_traj_sensor_measurements_and_params[test_indices]
            # print(train_traj_sensor_measurements_and_params.shape[2])
            flattened_train_traj_sensor_measurements_and_params = train_traj_sensor_measurements_and_params.reshape(
                train_traj_sensor_measurements_and_params.shape[0] * train_traj_sensor_measurements_and_params.shape[1],
                train_traj_sensor_measurements_and_params.shape[2])
            
            flattened_valid_traj_sensor_measurements_and_params = valid_traj_sensor_measurements_and_params.reshape(
                valid_traj_sensor_measurements_and_params.shape[0] * valid_traj_sensor_measurements_and_params.shape[1],
                valid_traj_sensor_measurements_and_params.shape[2])
            
            flattened_test_traj_sensor_measurements_and_params = test_traj_sensor_measurements_and_params.reshape(
                test_traj_sensor_measurements_and_params.shape[0] * test_traj_sensor_measurements_and_params.shape[1],
                test_traj_sensor_measurements_and_params.shape[2])
            # print('flattened_train_traj_sensor_measurements_and_params:', flattened_train_traj_sensor_measurements_and_params.shape)
            # print('flattened_valid_traj_sensor_measurements_and_params:', flattened_valid_traj_sensor_measurements_and_params.shape)
            flattened_all_traj_sensor_measurements_and_params = all_traj_sensor_measurements_and_params.reshape(
                all_traj_sensor_measurements_and_params.shape[0] * all_traj_sensor_measurements_and_params.shape[1],
                all_traj_sensor_measurements_and_params.shape[2])
            # fit
            # self.fit_sensors(self.data[i].shape[0], method) # use all time indices of train trajectories
            # self.fit_sensors(len(self.time), method) # use all time indices of train trajectories
            self.fit_sensors(flattened_train_traj_sensor_measurements_and_params)
            # print('flattened_train_traj_sensor_measurements_and_params:', flattened_train_traj_sensor_measurements_and_params.shape)
            # transform
            transformed_flattened_train_traj_sensor_measurements_and_params, transformed_flattened_valid_traj_sensor_measurements_and_params,transformed_flattened_test_traj_sensor_measurements_and_params = \
            self.transform_sensor(flattened_train_traj_sensor_measurements_and_params, flattened_valid_traj_sensor_measurements_and_params,
                                                        flattened_test_traj_sensor_measurements_and_params)
            # print('transformed_flattened_train_traj_sensor_measurements_and_params:', transformed_flattened_train_traj_sensor_measurements_and_params.shape)
            # print('ntimes', self.ntimes)
            # reshape back to (n_traj, n_time, n_sensors + n_params) to simply lag generating process
            transformed_train_traj_sensor_measurements_and_params = transformed_flattened_train_traj_sensor_measurements_and_params.reshape(
                int(transformed_flattened_train_traj_sensor_measurements_and_params.shape[0]/self.ntimes), self.ntimes, self.nsensors + self.nparams)
            transformed_valid_traj_sensor_measurements_and_params = transformed_flattened_valid_traj_sensor_measurements_and_params.reshape(
                int(transformed_flattened_valid_traj_sensor_measurements_and_params.shape[0]/self.ntimes), self.ntimes, self.nsensors + self.nparams)
            transformed_test_traj_sensor_measurements_and_params = transformed_flattened_test_traj_sensor_measurements_and_params.reshape(
                int(transformed_flattened_test_traj_sensor_measurements_and_params.shape[0]/self.ntimes), self.ntimes, self.nsensors + self.nparams)
            # generate X data
            X_train, X_valid, X_test = self.generate_X(transformed_train_traj_sensor_measurements_and_params, transformed_valid_traj_sensor_measurements_and_params,
                                                        transformed_test_traj_sensor_measurements_and_params)
    
        # parametric case

        # print('X_train:', X_train.shape)
        # print('X_valid:', X_valid.shape)
        # print('X_test:', X_test.shape)
        
        # Processing regarding Y
        # flattens full state data into into 2D array with time along axis 0.
        train_data = self.data[train_indices]
        val_data = self.data[val_indices]
        test_data = self.data[test_indices]
        train_data_flattened = None
        valid_data_flattened = None
        test_data_flattened = None
        for i in range(train_data.shape[0]):
            data = train_data[i]
            if train_data_flattened is None:
                train_data_flattened = self.flatten(data)
            else:
                train_data_flattened = np.vstack((train_data_flattened, self.flatten(data)))
        
        for i in range(val_data.shape[0]):
            data = val_data[i]
            if valid_data_flattened is None:
                valid_data_flattened = self.flatten(data)
            else:
                valid_data_flattened = np.vstack((valid_data_flattened, self.flatten(data)))

        for i in range(test_data.shape[0]):
            data = test_data[i]
            if test_data_flattened is None:
                test_data_flattened = self.flatten(data)
            else:
                test_data_flattened = np.vstack((test_data_flattened, self.flatten(data)))

        # fit (fit and transform can be combine with a wrapper or just integrate the code together)
        self.fit(train_data_flattened)
        # transform
        y_train, y_valid, y_test = self.transform(train_data_flattened, valid_data_flattened, test_data_flattened)
        self.Y_spatial_dim = y_train.shape[1]
        # generate y data
        # y_train, y_valid, y_test = self.generate_y(train_indices, val_indices, test_indices, method)
        self.full_state_data = None
        print('done generating dataset')
        # print('self.sensor_measurements_pd',self.sensor_measurements_pd)
        # full_state_data = self.unflatten(full_state_data)
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


    def fit_sensors(self, data):
        """
        Takes in train_indices
        Expects self.sensor_measurements to be a 2D nunpy array with time on axis 0.
        Scaling: fits either MinMaxScaler or Standard Scaler.
        Stores fitted scalers as object attributes.
        """
        # scaling full-state data
        # TODO: what if self.scaling is None?
        if self.scaling is not None:
            scaler_class = MinMaxScaler if self.scaling == 'minmax' else StandardScaler
            scaler = scaler_class()
            self.sensor_scaler = scaler.fit(data)


    def transform_sensor(self, train, valid, test):
        """
        Expects self.sensor_measurements to be a 2D nunpy array with time on axis 0.
        self.transformed_sensor_data to scaled scnsor_data (optional) have time on axis 0 (transpose).
        """
        # Transpose compressed data, time on axis 0
        # transformed_sensor_data = self.sensor_measurements.T
        # Perform scaling if all scaler-related attributes exist
        if self.sensor_scaler is not None:
            train = self.sensor_scaler.transform(train)
            valid = self.sensor_scaler.transform(valid)
            test = self.sensor_scaler.transform(test)
        return train, valid, test

    def fit_transform(self, train_indices, method):
        pass
    
    def fit(self, train_data):
        """
        Input: train_indices and method ("random" or "sequential")
        Expects self.data to be flattened with time on axis 0.
        Compression: fits standard scaler and save left singular values and singular values
        Scaling: fits either MinMaxScaler or Standard Scaler.
        Stores fitted scalers as object attributes.
        """
        V_component = None
        # compression
        if self.n_components is not None:
            # standard scale data
            sc = StandardScaler()
            sc.fit(train_data)
            self.scaler_before_svd = sc
            train_data_std_scaled = sc.transform(train_data)
            # rSVD
            # print('train_data_std_scaled.shape should be ntimes * ntraj * train_size',train_data_std_scaled.shape)
            U, S, V = randomized_svd(train_data_std_scaled, n_components=self.n_components, n_iter='auto')

            # TODO fit compression
            self.right_singular_values = V
            self.left_singular_values = U
            self.singular_values = S
            compressed_train_data = train_data_std_scaled @ V.transpose()
            print("compressed full_state_data:", compressed_train_data.shape)

        # transpose self.full_state_data so time on axis 0
        # self.full_state_data = self.full_state_data.T
        # transpose V_component so time on axis 0
        # V_component = V_component.T


        # scaling full-state data
        if self.scaling is not None:
            scaler_class = MinMaxScaler if self.scaling == 'minmax' else StandardScaler
            scaler = scaler_class()
            if self.n_components is not None:
                self.scaler = scaler.fit(compressed_train_data)
            else:
                self.scaler = scaler.fit(compressed_train_data)


    def transform(self, train_data, valid_data, test_data):
        """
        Expects self.full_state_data to be flattened with time on axis 0.
        Generates transfomed data which is self.full_state_data compressed (optional) and scaled (optional).
        """
        # Perform compression if all compression-related attributes exist
        if self.right_singular_values is not None and self.scaler_before_svd is not None:
            train_transformed_data = self.scaler_before_svd.transform(train_data)
            valid_transformed_data = self.scaler_before_svd.transform(valid_data)
            test_transformed_data = self.scaler_before_svd.transform(test_data)
            # TODO: transformed_data now shape (t, nstate) DAVID LOOK HERE
            # use matteo's

            train_transformed_data = train_transformed_data @ np.transpose(self.right_singular_values)
            valid_transformed_data = valid_transformed_data @ np.transpose(self.right_singular_values)
            test_transformed_data = test_transformed_data @ np.transpose(self.right_singular_values)

            # s_matrix = np.diag(self.singular_values[method]) # diagonal matrix of singular values
            # s_inv = np.linalg.inv(s_matrix) # compute inverse of singular values matrix
            # transformed_data = np.dot(s_inv, np.dot(self.left_singular_values.get(method).T, transformed_data)) # calculate V_T

        # Transpose compressed data, time on axis 0
        # transformed_data = transformed_data.T

        # Perform scaling if all scaler-related attributes exist
        if self.scaler is not None:
            train_transformed_data = self.scaler.transform(train_transformed_data)
            valid_transformed_data = self.scaler.transform(valid_transformed_data)
            test_transformed_data = self.scaler.transform(test_transformed_data)
        return train_transformed_data, valid_transformed_data, test_transformed_data


    def generate_X(self, train_data, valid_data, test_data):
        """
        Generates the input data for SHRED.
        Expects self.sensor_measurements to be a 2D numpy array with time is axis 0.
        Output: 3D torch.tensor with timesteps along axis 0, lags along axis 1, sensors along axis 2.
        Output: 3D numpy arrays with timesteps along axis 0, lags along axis 1, sensors along axis 2.
        """
        # (n_traj, n_time, n_sensors + n_params)
        # need to pass in the first lags number of data set well though
        train_lagged_sequences = None
        for i in range(train_data.shape[0]):
            data = train_data[i]
            if train_lagged_sequences is None:
                train_lagged_sequences = generate_lagged_sequences_from_sensor_measurements(data, self.lags)
            else:
                train_lagged_sequences = np.concatenate((train_lagged_sequences, generate_lagged_sequences_from_sensor_measurements(data, self.lags)), 0)

        valid_lagged_sequences = None
        for i in range(valid_data.shape[0]):
            data = valid_data[i]
            if valid_lagged_sequences is None:
                valid_lagged_sequences = generate_lagged_sequences_from_sensor_measurements(data, self.lags)
            else:
                valid_lagged_sequences = np.concatenate((valid_lagged_sequences, generate_lagged_sequences_from_sensor_measurements(data, self.lags)), 0)

        test_lagged_sequences = None
        for i in range(test_data.shape[0]):
            data = test_data[i]
            if test_lagged_sequences is None:
                test_lagged_sequences = generate_lagged_sequences_from_sensor_measurements(data, self.lags)
            else:
                test_lagged_sequences = np.concatenate((test_lagged_sequences, generate_lagged_sequences_from_sensor_measurements(data, self.lags)), 0)

        return train_lagged_sequences, valid_lagged_sequences, test_lagged_sequences
    
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


    def inverse_transform(self, data, uncompress, unscale):
        """
        Expects data to be a np array with time on axis 0.
        (Field specific output fron SHRED/Reconstructor)
        """
        # check if scaler fitted on reconstructor is not None
        if self.scaler is not None and unscale is True:
            data = self.scaler.inverse_transform(data)
        # check if compression is None
        if self.right_singular_values is not None and uncompress is True:
            data = data @ self.right_singular_values
        return data
    

    def generate_X_New(self,data):
        # (n_traj, n_time, n_sensors + n_params)
        # need to pass in the first lags number of data set well though
        lagged_sequences = None
        for i in range(data.shape[0]):
            data = data[i]
            if lagged_sequences is None:
                lagged_sequences = generate_lagged_sequences_from_sensor_measurements(data, self.lags)
            else:
                lagged_sequences = np.concatenate((lagged_sequences, generate_lagged_sequences_from_sensor_measurements(data, self.lags)), 0)

        # timesteps = end+1 # end_time inclusive
        # nsensors = self.sensor_measurements.shape[1] # make sure measurements dim matches
        # complete_measurements = np.full((timesteps, nsensors), np.nan)
        # if timesteps > len(self.sensor_measurements):
        #     print('hello')
        #     complete_measurements[0:len(self.sensor_measurements),:] = self.sensor_measurements
        # else:
        #     print('bye')
        #     complete_measurements[0:timesteps,:] = self.sensor_measurements[0:timesteps,:]
        # if measurements is not None and time is not None:
        #     for i in range(len(time)):
        #         if time[i] < complete_measurements.shape[0]:
        #             complete_measurements[time[i],:] = measurements[i,:]
        complete_measurements = self.sensor_scaler['random'].transform(complete_measurements)
        return complete_measurements

    # used for transforming raw new sensor measurements
    def transform_X(self, measurements):
        return self.sensor_scaler['random'].transform(measurements)
