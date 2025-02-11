from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.extmath import randomized_svd
from .utils import *

class ParametricSHREDDataProcessor:
    """
    ParametricSHREDDataProcessor manages a single dataset.
    ParametricSHREDDataProcessor objects are created by the `add` method in ParametricSHREDDataManager.
    """

    def __init__(self, data, random_sensors, stationary_sensors, mobile_sensors, lags, compression, params, id):
        """
        Inputs:
        - data: file path to a .npz file (string) or numpy array of shape (ntrajectories, ntime, x1, x2, x3,...,xN) where x is spatial dimensions
        - random_sensors: number of randomly placed stationary sensors (integer).
        - stationary_sensors: coordinates of stationary sensors. Each sensor coordinate is a tuple.
                              If multiple stationary sensors, put tuples into a list (tuple or list of tuples).
        - mobile_sensors: list of coordinates (tuple) for a mobile sensor (length of list should match number of timesteps in `data`).
                          If multiple mobile_sensors, use a nested list (list of tuples, or nested list of tuples).
        - time: 1D numpy array of timestamps
        - lags: number of time steps to look back (integer).
        - compression: dimensionality reduction (boolean or integer).
        - id: unique identifier for the dataset (string).
        - lags: number of time steps to look back (integer).
        - compression: dimensionality reduction (boolean or integer).
        - params: dataset of parameters of shape (ntrajectories, ntimes, nparams)
        - id: unique identifier for the dataset (string).
        """
        # Generic
        if compression == True:
            self.n_components = 50
        elif compression == False:
            self.n_components = None
        elif isinstance(compression, int):
            self.n_components = compression
        self.random_sensors = random_sensors # number of randomly placed sensors
        self.stationary_sensors = stationary_sensors # stationary sensor locations
        self.mobile_sensors = mobile_sensors # mobile sensor locations
        self.lags = lags # number of timesteps to look back
        self.compression = compression
        self.sensor_scaler = {}
        self.scaler_before_svd = {}
        self.left_singular_values = {}
        self.singular_values = {}
        self.right_singular_values = {}
        self.scaler = {} # stores scaler of full-state data
        self.sensor_summary = None
        self.sensor_measurements = None
        self.sensor_measurements_pd = None
        self.id = str(id)
        self.Y_spatial_dim = None
        # Specific to ParametricSHREDDataProcessor
        self.full_state_param_data = get_data(data)
        self.data_spatial_shape = self.full_state_param_data.shape[2:]
        self.ntrajectories = self.full_state_param_data.shape[0]
        self.ntimes = self.full_state_param_data.shape[1]
        if params is not None:
            self.nparams = params.shape[2]
        else:
            self.nparams = 0
        self.params = params
        self.params_pd = None

        if self.random_sensors is not None or self.stationary_sensors is not None or self.mobile_sensors is not None:
            random_sensor_locations = None
            # for each trajectory
            for i in range(self.full_state_param_data.shape[0]):
                full_state_data = self.full_state_param_data[i]
                # calculate random sensors once
                if i == 0:
                    if self.random_sensors is not None:
                        random_sensor_locations = generate_random_sensor_locations(full_state = full_state_data, num_sensors = self.random_sensors)
                sensor_measurements_dict = get_sensor_measurements(full_state_data=full_state_data,
                                                                id = self.id, time = None,
                                                                stationary_sensors=self.stationary_sensors,
                                                                mobile_sensors=self.mobile_sensors,
                                                                random_sensors=random_sensor_locations) # get sensor data
                sensor_measurements = sensor_measurements_dict['sensor_measurements'].to_numpy()[np.newaxis, :] # add trajectory axis
                sensor_measurements_dict['sensor_measurements'].insert(0, "trajectory", i) # add trajectory column
                sensor_measurements_pd = sensor_measurements_dict['sensor_measurements']
                sensor_summary = sensor_measurements_dict['sensor_summary']
                if self.sensor_measurements is None:
                    self.sensor_measurements = sensor_measurements
                else:
                    self.sensor_measurements = np.concatenate((self.sensor_measurements, sensor_measurements), axis = 0)
                if self.sensor_measurements_pd is None:
                    self.sensor_measurements_pd = sensor_measurements_pd
                else:
                    self.sensor_measurements_pd = pd.concat([self.sensor_measurements_pd, sensor_measurements_pd], axis=0, ignore_index=True)
                if self.sensor_summary is None:
                    self.sensor_summary = sensor_summary

        self.nsensors = self.sensor_measurements.shape[2]

        if self.params is not None:
            param_dfs_list = []
            for i in range(self.params.shape[0]):
                df = pd.DataFrame(
                    self.params[i],
                    columns=[f"{self.id} param {j}" for j in range(self.params.shape[2])]
                )
                df.insert(0, "trajectory", i)
                param_dfs_list.append(df)
            self.params_pd = pd.concat(param_dfs_list, axis=0, ignore_index=True)


    def generate_dataset(self, train_indices, val_indices, test_indices):
        """
        Sets train, val, and test SHREDDataset objects with generated dataset.
        """
        X_train, X_val, X_test = None, None, None
        y_train, y_val, y_test = None, None, None
        # Generate X
        if self.sensor_measurements is not None:
            if self.params is not None:
                inputs = np.concatenate((self.sensor_measurements, self.params), axis=2)  # Shape (ntrajs, ntimes, nsensors + nparams)
            else:
                inputs = self.sensor_measurements

            train_inputs = inputs[train_indices]
            val_inputs = inputs[val_indices]
            test_inputs = inputs[test_indices]

            # Helper lambda to flatten data: shape (n_traj, n_time, n_features) -> (n_traj*n_time, n_features)
            flattened_train_inputs = train_inputs.reshape(-1, train_inputs.shape[-1])
            flattened_val_inputs   = val_inputs.reshape(-1, val_inputs.shape[-1])
            flattened_test_inputs  = test_inputs.reshape(-1, test_inputs.shape[-1])

            # flattened_train_inputs_inputs = train_inputs.reshape(
            #     train_inputs.shape[0] * train_inputs.shape[1],
            #     train_inputs.shape[2]) # Shape (ntajs * ntimes, nsensors + nparams)
            
            # flattened_val_inputs_inputs = val_inputs.reshape(
            #     val_inputs.shape[0] * val_inputs.shape[1],
            #     val_inputs.shape[2])
            
            # flattened_test_inputs_inputs = test_inputs.reshape(
            #     test_inputs.shape[0] * test_inputs.shape[1],
            #     test_inputs.shape[2])

            # Fit sensors on the training data and transform all datasets
            scaler = MinMaxScaler()
            self.sensor_scaler = scaler.fit(flattened_train_inputs)

            transformed_flat_train = self.sensor_scaler.transform(flattened_train_inputs)
            transformed_flat_val = self.sensor_scaler.transform(flattened_val_inputs)
            transformed_flat_test = self.sensor_scaler.transform(flattened_test_inputs)

            # self.fit_sensors(flattened_train_inputs_inputs)
            # transformed_flattened_train_inputs_inputs, transformed_flattened_val_inputs_inputs,transformed_flattened_test_inputs_inputs = \
            # self.transform_sensor(flattened_train_inputs_inputs, flattened_val_inputs_inputs,
            #                                             flattened_test_inputs_inputs)

            # Reshape back to (n_traj, n_time, nsensors + nparams)
            transformed_train = transformed_flat_train.reshape(-1, self.ntimes, self.nsensors + self.nparams)
            transformed_val = transformed_flat_val.reshape(-1, self.ntimes, self.nsensors + self.nparams)
            transformed_test = transformed_flat_test.reshape(-1, self.ntimes, self.nsensors + self.nparams)
            
            # reshape back to (n_traj, n_time, n_sensors + n_params) to simply lag generating process
            # transformed_train_inputs = transformed_flattened_train_inputs_inputs.reshape(
            #     int(transformed_flattened_train_inputs_inputs.shape[0]/self.ntimes), self.ntimes, self.nsensors + self.nparams)
            # transformed_val_inputs = transformed_flattened_val_inputs_inputs.reshape(
            #     int(transformed_flattened_val_inputs_inputs.shape[0]/self.ntimes), self.ntimes, self.nsensors + self.nparams)
            # transformed_test_inputs = transformed_flattened_test_inputs_inputs.reshape(
            #     int(transformed_flattened_test_inputs_inputs.shape[0]/self.ntimes), self.ntimes, self.nsensors + self.nparams)

            def generate_sequences(dataset):
                # dataset is expected to be of shape (n_trajectories, n_time, n_features)
                sequences = [
                    generate_lagged_sequences_from_sensor_measurements(traj_data, self.lags)
                    for traj_data in dataset
                ]
                return np.concatenate(sequences, axis=0)

            # Generate lagged sequences for train, validation, and test datasets
            X_train = generate_sequences(transformed_train)
            X_val = generate_sequences(transformed_val)
            X_test = generate_sequences(transformed_test)


            # X_train = None
            # for i in range(transformed_train_inputs.shape[0]):
            #     data = transformed_train_inputs[i]
            #     if X_train is None:
            #         X_train = generate_lagged_sequences_from_sensor_measurements(data, self.lags)
            #     else:
            #         X_train = np.concatenate((X_train, generate_lagged_sequences_from_sensor_measurements(data, self.lags)), 0)

            # X_val = None
            # for i in range(transformed_val_inputs.shape[0]):
            #     data = transformed_val_inputs[i]
            #     if X_val is None:
            #         X_val = generate_lagged_sequences_from_sensor_measurements(data, self.lags)
            #     else:
            #         X_val = np.concatenate((X_val, generate_lagged_sequences_from_sensor_measurements(data, self.lags)), 0)

            # X_test = None
            # for i in range(transformed_test_inputs.shape[0]):
            #     data = transformed_test_inputs[i]
            #     if X_test is None:
            #         X_test = generate_lagged_sequences_from_sensor_measurements(data, self.lags)
            #     else:
            #         X_test = np.concatenate((X_test, generate_lagged_sequences_from_sensor_measurements(data, self.lags)), 0)

                        # generate X data
            # X_train, X_val, X_test = self.generate_X(transformed_train_inputs, transformed_val_inputs,
            #                                             transformed_test_inputs)

        # parametric case

        # print('X_train:', X_train.shape)
        # print('X_val:', X_val.shape)
        # print('X_test:', X_test.shape)
        
        # Processing regarding Y
        # flattens full state data into into 2D array with time along axis 0.
        train_data = self.full_state_param_data[train_indices]
        val_data = self.full_state_param_data[val_indices]
        test_data = self.full_state_param_data[test_indices]
        train_data_flattened = train_data.reshape(-1, train_data.shape[-1])

        val_data_flattened   = val_data.reshape(-1, val_data.shape[-1])
        test_data_flattened  = test_data.reshape(-1, test_data.shape[-1])

        # Fit the model on the training data
        self.fit(train_data_flattened)

        # Transform the data; the transform function returns the processed versions for each split.
        y_train, y_val, y_test = self.transform(train_data_flattened, val_data_flattened, test_data_flattened)
        self.Y_spatial_dim = y_train.shape[1]
        # self.full_state_data = unflatten(data = self.full_state_data, spatial_shape=self.data_spatial_shape)


        # train_data_flattened = None
        # val_data_flattened = None
        # test_data_flattened = None
        # for i in range(train_data.shape[0]):
        #     data = train_data[i]
        #     if train_data_flattened is None:
        #         train_data_flattened = self.flatten(data)
        #     else:
        #         train_data_flattened = np.vstack((train_data_flattened, self.flatten(data)))
        
        # for i in range(val_data.shape[0]):
        #     data = val_data[i]
        #     if val_data_flattened is None:
        #         val_data_flattened = self.flatten(data)
        #     else:
        #         val_data_flattened = np.vstack((val_data_flattened, self.flatten(data)))

        # for i in range(test_data.shape[0]):
        #     data = test_data[i]
        #     if test_data_flattened is None:
        #         test_data_flattened = self.flatten(data)
        #     else:
        #         test_data_flattened = np.vstack((test_data_flattened, self.flatten(data)))

        # # fit (fit and transform can be combine with a wrapper or just integrate the code together)
        # self.fit(train_data_flattened)
        # # transform
        # y_train, y_val, y_test = self.transform(train_data_flattened, val_data_flattened, test_data_flattened)
        # self.Y_spatial_dim = y_train.shape[1]
        # generate y data
        # y_train, y_val, y_test = self.generate_y(train_indices, val_indices, test_indices, method)
        # self.full_state_data = None
        # print('done generating dataset')
        # print('self.sensor_measurements_pd',self.sensor_measurements_pd)
        # full_state_data = self.unflatten(full_state_data)
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }


    # def fit_sensors(self, data):
    #     """
    #     Takes in train_indices
    #     Expects self.sensor_measurements to be a 2D nunpy array with time on axis 0.
    #     Scaling: fits either MinMaxScaler or Standard Scaler.
    #     Stores fitted scalers as object attributes.
    #     """
    #     # scaling full-state data
    #     # TODO: what if self.scaling is None?
    #     if self.scaling is not None:
    #         scaler_class = MinMaxScaler if self.scaling == 'minmax' else StandardScaler
    #         scaler = scaler_class()
    #         self.sensor_scaler = scaler.fit(data)


    # def transform_sensor(self, train, val, test):
    #     """
    #     Expects self.sensor_measurements to be a 2D nunpy array with time on axis 0.
    #     self.transformed_sensor_data to scaled scnsor_data (optional) have time on axis 0 (transpose).
    #     """
    #     # Transpose compressed data, time on axis 0
    #     # transformed_sensor_data = self.sensor_measurements.T
    #     # Perform scaling if all scaler-related attributes exist
    #     if self.sensor_scaler is not None:
    #         train = self.sensor_scaler.transform(train)
    #         val = self.sensor_scaler.transform(val)
    #         test = self.sensor_scaler.transform(test)
    #     return train, val, test

    # def fit_transform(self, train_indices, method):
    #     pass
    
    def fit(self, train_data):
        """
        Input: train_indices and method ("random" or "sequential")
        Expects self.full_state_param_data to be flattened with time on axis 0.
        Compression: fits standard scaler and save left singular values and singular values
        Scaling: fits either MinMaxScaler or Standard Scaler.
        Stores fitted scalers as object attributes.
        """
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
            self.right_singular_values = V
            self.left_singular_values = U
            self.singular_values = S
            compressed_train_data = train_data_std_scaled @ V.transpose()
        scaler = MinMaxScaler()
        if self.n_components is not None:
            self.scaler = scaler.fit(compressed_train_data)
        else:
            self.scaler = scaler.fit(train_data)


    def transform(self, train_data, val_data, test_data):
        """
        Expects self.full_state_data to be flattened with time on axis 0.
        Generates transfomed data which is self.full_state_data compressed (optional) and scaled (optional).
        """
        # Perform compression if all compression-related attributes exist
        if self.right_singular_values is not None and self.scaler_before_svd is not None:
            train_transformed_data = self.scaler_before_svd.transform(train_data)
            val_transformed_data = self.scaler_before_svd.transform(val_data)
            test_transformed_data = self.scaler_before_svd.transform(test_data)
            # TODO: transformed_data now shape (t, nstate) DAVID LOOK HERE
            # use matteo's

            train_transformed_data = train_transformed_data @ np.transpose(self.right_singular_values)
            val_transformed_data = val_transformed_data @ np.transpose(self.right_singular_values)
            test_transformed_data = test_transformed_data @ np.transpose(self.right_singular_values)

            # s_matrix = np.diag(self.singular_values[method]) # diagonal matrix of singular values
            # s_inv = np.linalg.inv(s_matrix) # compute inverse of singular values matrix
            # transformed_data = np.dot(s_inv, np.dot(self.left_singular_values.get(method).T, transformed_data)) # calculate V_T

        # Transpose compressed data, time on axis 0
        # transformed_data = transformed_data.T

        # Perform scaling if all scaler-related attributes exist
        if self.scaler is not None:
            train_transformed_data = self.scaler.transform(train_transformed_data)
            val_transformed_data = self.scaler.transform(val_transformed_data)
            test_transformed_data = self.scaler.transform(test_transformed_data)
        return train_transformed_data, val_transformed_data, test_transformed_data


    # def generate_X(self, train_data, val_data, test_data):
    #     """
    #     Generates the input data for SHRED.
    #     Expects self.sensor_measurements to be a 2D numpy array with time is axis 0.
    #     Output: 3D torch.tensor with timesteps along axis 0, lags along axis 1, sensors along axis 2.
    #     Output: 3D numpy arrays with timesteps along axis 0, lags along axis 1, sensors along axis 2.
    #     """
    #     # (n_traj, n_time, n_sensors + n_params)
    #     # need to pass in the first lags number of data set well though
    #     train_lagged_sequences = None
    #     for i in range(train_data.shape[0]):
    #         data = train_data[i]
    #         if train_lagged_sequences is None:
    #             train_lagged_sequences = generate_lagged_sequences_from_sensor_measurements(data, self.lags)
    #         else:
    #             train_lagged_sequences = np.concatenate((train_lagged_sequences, generate_lagged_sequences_from_sensor_measurements(data, self.lags)), 0)

    #     val_lagged_sequences = None
    #     for i in range(val_data.shape[0]):
    #         data = val_data[i]
    #         if val_lagged_sequences is None:
    #             val_lagged_sequences = generate_lagged_sequences_from_sensor_measurements(data, self.lags)
    #         else:
    #             val_lagged_sequences = np.concatenate((val_lagged_sequences, generate_lagged_sequences_from_sensor_measurements(data, self.lags)), 0)

    #     test_lagged_sequences = None
    #     for i in range(test_data.shape[0]):
    #         data = test_data[i]
    #         if test_lagged_sequences is None:
    #             test_lagged_sequences = generate_lagged_sequences_from_sensor_measurements(data, self.lags)
    #         else:
    #             test_lagged_sequences = np.concatenate((test_lagged_sequences, generate_lagged_sequences_from_sensor_measurements(data, self.lags)), 0)

    #     return train_lagged_sequences, val_lagged_sequences, test_lagged_sequences
    
    # def generate_y(self, train_indices, val_indices, test_indices, method):
    #     """
    #     Generates the target data for SHRED.
    #     The target data are full-state data that is compresssed (optional) and scaled (optional),
    #     and flattens the state data.
    #     Output: 2D numpy array with timesteps along axis 0 and flattened state data along axis 1.
    #     """
    #     # train = self.transformed_data[method][train_indices + self.lags]
    #     # val = self.transformed_data[method][val_indices + self.lags]
    #     # test = self.transformed_data[method][test_indices + self.lags]
    #     train = self.transformed_data[method][train_indices]
    #     val = self.transformed_data[method][val_indices]
    #     test = self.transformed_data[method][test_indices]
    #     return train, val, test
    
    # def flatten(self, data):
    #     """
    #     Takes in a nd array where the time is along the axis 0.
    #     Flattens the nd array into 2D array with time along axis 0.
    #     """
    #     self.original_shape = data.shape
    #     # Reshape the data: keep time (axis 0) and flatten the remaining dimensions
    #     return data.reshape(data.shape[0], -1)
    
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
        self.full_state_param_data = None


    def inverse_transform(self, data, uncompress):
        """
        Expects data to be a np array with time on axis 0.
        (Field specific output fron SHRED/Reconstructor)
        """
        # check if scaler fitted on reconstructor is not None
        if self.scaler is not None:
            data = self.scaler.inverse_transform(data)
        # check if compression is None
        if self.right_singular_values is not None and uncompress is True:
            data = data @ self.right_singular_values
            data = self.scaler_before_svd.inverse_transform(data)
            original_shape = (-1, self.ntimes) + self.data_spatial_shape
            data = data.reshape(original_shape)
        return data
    

    def generate_X(self,data):
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
