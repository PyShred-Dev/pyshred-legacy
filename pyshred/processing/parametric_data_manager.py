import torch
from .utils import * 
from .parametric_data_processor import *

class ParametricSHREDDataManager:
    """
    ParametricSHREDDataManager is the orchestrator of ParametricSHREDDataProcessor objects.
    
    Methods
    ----------
    - add: Adds a new ParametricSHREDDataProcessor object to the ParametricSHREDDataManager.
    - preprocess: Generate SHREDDataset objects for training, validation, and testing.
    - postprocess: Postprocess the output data by applying inverse transformations to each dataset field.
    """

    # TODO: Recon, Predict, Forecast, will need sensor summary to notify user order of sensor measurements
    # TODO: sensor summary will also need to include parameter summary
    # TODO: each sensor and parameter can be identified to a dataset/field/regime with the unique id of the SHREDDataProcessor


    def __init__(self, lags = 20, train_size = 0.8, val_size = 0.1, test_size = 0.1, compression = True):
        # Generic
        self.compression = compression # boolean or int
        self.lags = lags # number of time steps to look back (int)
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.sensor_summary = None #
        self.sensor_measurements = None
        self.reconstructor_indices = None # a dict w/ 'train', 'val', 'test' indices for reconstructor
        self.reconstructor = [] # stores reconstructor datasets of each field
        # Specific to ParametricSHREDDataManager
        self.parametric_data_processors = [] # a list storing ParametricSHREDDataProcessor objects
        self.params = None

    #TODO: allow for sensor measurments directly as well
    def add(self, data, id, random_sensors = None, stationary_sensors = None, mobile_sensors = None, params=None, compression=None):
        """
        Creates and adds a new SHREDDataProcessor object.
        - file path: file path
        - random_sensors: number of randomly placed stationary sensors (integer).
        - stationary_sensors: coordinates of stationary sensors. Each sensor coordinate is a tuple.
                              If multiple stationary sensors, put tuples into a list (tuple or list of tuples).
        - mobile_sensors: list of coordinates (tuple) for a mobile sensor (length of list should match number of timesteps in `data`).
                          If multiple mobile_sensors, use a nested list (list of tuples, or nested list of tuples).
        - time: 1D numpy array of timestamps or list of 1D numpy arrays (if multiple trajectories)
        - lags: number of time steps to look back (integer).
        - compression: dimensionality reduction (boolean or integer).
        - scaling: scaling settings ('minmax', 'standard').
        - id: unique identifier for the dataset (string).
        """
        # data = get_data(data) # (n_traj, n_time, n_state)
        compression = compression if compression is not None else self.compression


        # create and initialize ParametricSHREDDataProcessor
        data_processor = ParametricSHREDDataProcessor(
            data=data,
            params = params, # add method dependent, might add data from a field without params?
            random_sensors=random_sensors,
            stationary_sensors=stationary_sensors,
            mobile_sensors=mobile_sensors,
            lags=self.lags,
            compression=compression,
            id=id,
        )

        # record sensor_summay and sensor_measurements
        if data_processor.sensor_summary is not None and data_processor.sensor_measurements_pd is not None:
            if self.sensor_summary is None and self.sensor_measurements is None:
                self.sensor_summary = data_processor.sensor_summary
                self.sensor_measurements = data_processor.sensor_measurements_pd
            else:
                self.sensor_summary = pd.concat([self.sensor_summary, data_processor.sensor_summary], axis = 0).reset_index(drop=True)
                # self.sensor_measurements = pd.merge(self.sensor_measurements, data_processor.sensor_measurements_pd, on=['trajectory'], how = 'inner')
                self.sensor_measurements = pd.concat(
                    [self.sensor_measurements.reset_index(drop=True), data_processor.sensor_measurements_pd.drop("trajectory", axis=1).reset_index(drop=True)],
                    axis=1
                )


        # record param_summay and param_measurements
        if data_processor.params_pd is not None:
            if self.params is None:
                self.params = data_processor.params_pd
            else:
                # self.params = pd.merge(self.params, data_processor.params_pd, on=['trajectory'], how = 'inner')
                self.params = pd.concat(
                    [self.params.reset_index(drop=True), data_processor.params_pd.drop("trajectory", axis=1).reset_index(drop=True)],
                    axis=1
                )


        # generate train/val/test indices
        if len(self.parametric_data_processors) == 0:
            self.reconstructor_indices = get_train_val_test_indices(data.shape[0], self.train_size, self.val_size, self.test_size, method = "random") # split randomly by trajectories

        dataset_dict = data_processor.generate_dataset(
            self.reconstructor_indices['train'],
            self.reconstructor_indices['val'],
            self.reconstructor_indices['test'],
        )
        self.reconstructor.append(dataset_dict)

        data_processor.discard_data()
        self.parametric_data_processors.append(data_processor)
        
    

    def preprocess(self):
        """
        Generates train, val, and test SHREDDataset objects.
        """

        def concatenate_datasets(existing_X, existing_y, new_X, new_y):
            """
            Helper function to concatenate datasets along the last axis, handling None values.
            """
            combined_X = None
            combined_y = None

            if existing_X is None:
                combined_X = new_X
            elif new_X is None:
                combined_X = existing_X
            else:
                combined_X = np.concatenate((existing_X, new_X), axis=-1)
            
            if existing_y is None:
                combined_y = new_y
            elif new_y is None:
                combined_y = existing_y
            else:
                combined_y = np.concatenate((existing_y, new_y), axis=-1)

            return (
                combined_X,
                combined_y,
            )

        # Initialize datasets for reconstructor and forecastor
        X_train_recon, y_train_recon = None, None
        X_valid_recon, y_valid_recon = None, None
        X_test_recon, y_test_recon = None, None

        # X_train_forecast, y_train_forecast = None, None
        # X_valid_forecast, y_valid_forecast = None, None
        # X_test_forecast, y_test_forecast = None, None

        # Process reconstructor datasets
        for dataset_dict in self.reconstructor:
            X_train_recon, y_train_recon = concatenate_datasets(
                X_train_recon, y_train_recon, dataset_dict['train'][0], dataset_dict['train'][1]
            )
            X_valid_recon, y_valid_recon = concatenate_datasets(
                X_valid_recon, y_valid_recon, dataset_dict['val'][0], dataset_dict['val'][1]
            )
            X_test_recon, y_test_recon = concatenate_datasets(
                X_test_recon, y_test_recon, dataset_dict['test'][0], dataset_dict['test'][1]
            )

        # Create TimeSeriesDataset and SHREDDataset objects
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert data to torch tensors and move to the specified device
        X_train_recon = torch.tensor(X_train_recon, dtype=torch.float32, device=device)
        y_train_recon = torch.tensor(y_train_recon, dtype=torch.float32, device=device)
        X_valid_recon = torch.tensor(X_valid_recon, dtype=torch.float32, device=device)
        y_valid_recon = torch.tensor(y_valid_recon, dtype=torch.float32, device=device)
        X_test_recon = torch.tensor(X_test_recon, dtype=torch.float32, device=device)
        y_test_recon = torch.tensor(y_test_recon, dtype=torch.float32, device=device)

        # X_train_forecast = torch.tensor(X_train_forecast, dtype=torch.float32, device=device)
        # y_train_forecast = torch.tensor(y_train_forecast, dtype=torch.float32, device=device)
        # X_valid_forecast = torch.tensor(X_valid_forecast, dtype=torch.float32, device=device)
        # y_valid_forecast = torch.tensor(y_valid_forecast, dtype=torch.float32, device=device)
        # X_test_forecast = torch.tensor(X_test_forecast, dtype=torch.float32, device=device)
        # y_test_forecast = torch.tensor(y_test_forecast, dtype=torch.float32, device=device)

        # Create TimeSeriesDataset objects
        train_recon_dataset = TimeSeriesDataset(X_train_recon, y_train_recon)
        valid_recon_dataset = TimeSeriesDataset(X_valid_recon, y_valid_recon)
        test_recon_dataset = TimeSeriesDataset(X_test_recon, y_test_recon)

        # train_forecast_dataset = TimeSeriesDataset(X_train_forecast, y_train_forecast)
        # valid_forecast_dataset = TimeSeriesDataset(X_valid_forecast, y_valid_forecast)
        # test_forecast_dataset = TimeSeriesDataset(X_test_forecast, y_test_forecast)

        SHRED_train_dataset = SHREDDataset(train_recon_dataset)
        SHRED_val_dataset = SHREDDataset(valid_recon_dataset)
        SHRED_test_dataset = SHREDDataset(test_recon_dataset)

        return SHRED_train_dataset, SHRED_val_dataset, SHRED_test_dataset
    



    def postprocess(self, data, postprocess = True, mode = "reconstruct"):
        results = {}
        start_index = 0
        for data_processor in self.parametric_data_processors:
            field_spatial_dim = data_processor.Y_spatial_dim
            field_data = data[:, start_index:start_index+field_spatial_dim]
            if isinstance(data, torch.Tensor):
                field_data = field_data.detach().cpu().numpy()
            start_index = field_spatial_dim + start_index
            field_data = data_processor.inverse_transform(field_data, postprocess)
            results[data_processor.id] = field_data
        return results
    

    def generate_X(self, measurements, params):
        """
        For each field (i.e., each ParametricSHREDDataProcessor), this method slices out
        the appropriate sensor measurements and parameter data, concatenates them along
        the last axis, and then calls the processor's generate_X method.
        
        Parameters
        ----------
        measurements : np.ndarray
            Array of sensor measurements with shape (n_traj, n_times, nsensors).
        params : np.ndarray
            Array of parameter measurements with shape (n_traj, n_times, nparams).
        
        Returns
        -------
        torch.Tensor
            Combined tensor of processed inputs with shape (n_traj, n_times, n_features)
            where n_features is the sum of the individual processed sensor and parameter features.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = None
        start_sensor = 0
        start_param = 0

        for data_processor in self.parametric_data_processors:
            # Determine the number of sensor features for this field.
            n_sensors = data_processor.sensor_measurements.shape[1]
            end_sensor = start_sensor + n_sensors

            # Determine the number of parameter features for this field, if any.
            if data_processor.params_pd is not None:
                n_params = data_processor.params_pd.shape[1]
            else:
                n_params = 0
            end_param = start_param + n_params

            # Slice out the sensor measurements for this field along the last axis.
            field_measurements = measurements[:, :, start_sensor:end_sensor]
            
            # If there are parameter measurements, slice and concatenate.
            if n_params > 0:
                field_params = params[:, :, start_param:end_param]
                # Concatenate along the last axis (features axis).
                combined_field_data = np.concatenate([field_measurements, field_params], axis=-1)
            else:
                combined_field_data = field_measurements

            # Let the data processor generate its X (this might include lagged sequences etc.).
            field_result = data_processor.generate_X(combined_field_data)
            # field_result should have shape (n_traj, n_times, n_features_field)

            # Accumulate results from each field.
            if results is None:
                results = field_result
            else:
                # Concatenate along the feature axis.
                results = np.concatenate((results, field_result), axis=-1)

            # Update start indices for the next field.
            start_sensor = end_sensor
            start_param = end_param

        # Convert final result into a torch tensor on the chosen device.
        results = torch.tensor(results, dtype=torch.float32, device=device)
        return results

    # def generate_X(self, measurements, params):
    #     # n_traj, n_times, nsensors
    #     # n_traj, n_times, nparams
    #     # for each field, concatenate the appropriate measurements with the appropriate params
    #     # seperate out params measurements from sensor measurements, so it is clear the order of params and which belong to which dataset
    #     results = None
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     start_sensor = 0
    #     for data_processor in self.parametric_data_processors:
    #         end_sensor = start_sensor + data_processor.sensor_measurements.shape[1]
    #         field_measurements = measurements[:,start_sensor:end_sensor]
    #         result = data_processor.generate_X(field_measurements) # (n_traj, n_time, n_sensors + n_params)



    #     results = torch.tensor(results, dtype=torch.float32, device=device)
    #     return results
        # start_sensor = 0
        # if start is None and end is None:
        #     # generate lagged sequences from measurements
        #     for data_processor in self.parametric_data_processors:
        #         end_sensor = start_sensor + data_processor.sensor_measurements.shape[1]
        #         result = data_processor.transform_X(measurements[:,start_sensor:end_sensor])
        #         if results is None:
        #             results = result
        #         else:
        #             results = np.concatenate((results, result), axis = 1)
        #         start_sensor = end_sensor
        #     results = generate_lagged_sequences_from_sensor_measurements(results, self.lags)
        # else:
        #     # generate lagged sequences from start to end (inclusive)
        #     for data_processor in self.parametric_data_processors:
        #         if data_processor.sensor_measurements is not None:
        #             end_sensor = start_sensor + data_processor.sensor_measurements.shape[1]
        #             if measurements is not None:
        #                 field_measurements = measurements[:,start_sensor:end_sensor]
        #                 result = data_processor.generate_X(end = end, measurements = field_measurements, time = time)
        #             else:
        #                 result = data_processor.generate_X(end = end, measurements = None, time = time)
        #             if results is None:
        #                 results = result
        #             else:
        #                 results = np.concatenate((results, result), axis = 1)
        #             start_sensor = end_sensor
        #     # extract indices without data (gaps)
        #     gap_indices = np.where(np.isnan(results).any(axis=1))[0]
        #     for gap in gap_indices:
        #         # gap is being forecasted, gap - 1 is 'current'
        #         gap_lagged_sequence = results[gap - 1 - self.lags:gap,:].copy()
        #         gap_lagged_sequence = gap_lagged_sequence[np.newaxis,:,:]
        #         gap_lagged_sequence = torch.tensor(gap_lagged_sequence, dtype=torch.float32, device=device)
        #         results[gap] = forecaster(gap_lagged_sequence).detach().numpy()
        #     results = generate_lagged_sequences_from_sensor_measurements(results, self.lags)
        #     results = results[start:end+1,:,:]
