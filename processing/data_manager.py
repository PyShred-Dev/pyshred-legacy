


import torch
from .utils import * 
from .data_processor import *

class SHREDDataManager:
    """
    SHREDDataManager is the orchestrator of SHREDDataProcessor objects.
    methods:
    - add: for creating SHREDData objects and adding to SHREDDataManager
    - remove: for removing SHREDData objects from SHREDDataManager (to be implemented)
    - preprocess: for generating train, validation, and test SHREDDataset objects
    - postprocess
    """

    def __init__(self, lags = 20, time = None, train_size = 0.75, val_size = 0.15, test_size = 0.15, scaling = "minmax", compression = True, reconstructor=True, forecastor=True):
        self.scaling = scaling
        self.compression = compression
        self.time = time # expects a 1D numpy array
        self.lags = lags # number of time steps to look back
        self.data_processors = [] # a list storing SHREDDataProcessor objects
        self.reconstructor_indices = None # a dict storing 'train', 'validation', 'test' indices for SHRED reconstructor
        self.forecastor_indices = None # a dict storing 'train', 'validation', 'test' indices for SHRED forecastor
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.reconstructor_data_elements = []
        self.forecastor_data_elements = []
        self.reconstructor_flag = reconstructor
        self.forecastor_flag = forecastor
        self.input_summary = None #
        self.sensor_measurements = None # all sensor measurement
        # self.reconstructor = reconstructor # flag for generating datasets for SHRED reconstructor
        # self.forecastor = forecastor # flag for generating datasets for SHRED forecaster

    def add_field(self, data, random_sensors = None, stationary_sensors = None, mobile_sensors = None, compression = True, id = None, scaling = "minmax", time = None):
        """
        Creates and adds a new SHREDDataProcessor object.
        - file path: file path to data (string)
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
        compression = compression if compression is not None else self.compression
        scaling = scaling if scaling is not None else self.scaling
        time = time if time is not None else self.time
        # generate train/val/test indices based on effective number of timesteps in initial SHREDDataProcessor object
        if len(self.data_processors) == 0:
            self.reconstructor_indices = get_train_val_test_indices(len(time), self.train_size, self.val_size, self.test_size, method = "random")
            self.forecastor_indices = get_train_val_test_indices(len(time), self.train_size, self.val_size, self.test_size, method = "sequential")
            # self.sensor_measurements = time.reshape(-1,1) #reshape 1D to 2D array of shape (ntime, 1)
        # create and initialize SHREDData object
        data_processor = SHREDDataProcessor(
            data=data,
            random_sensors=random_sensors,
            stationary_sensors=stationary_sensors,
            mobile_sensors=mobile_sensors,
            lags=self.lags,
            time=time,
            compression=compression,
            scaling=scaling,
            id=id
        )
        if self.reconstructor_flag:
            dataset_dict = data_processor.generate_dataset(
                        self.reconstructor_indices['train'],
                        self.reconstructor_indices['validation'],
                        self.reconstructor_indices['test'],
                        method='random'
                    )
            self.reconstructor_data_elements.append(dataset_dict)
        
        if self.forecastor_flag:
            dataset_dict = data_processor.generate_dataset(
                    self.forecastor_indices['train'],
                    self.forecastor_indices['validation'],
                    self.forecastor_indices['test'],
                    method='sequential'
                )
            self.forecastor_data_elements.append(dataset_dict)
        
        if data_processor.sensor_summary is not None and data_processor.sensor_measurements_pd is not None:
            if self.input_summary is None and self.sensor_measurements is None:
                self.input_summary = data_processor.sensor_summary
                self.sensor_measurements = data_processor.sensor_measurements_pd
            else:
                self.input_summary = pd.concat([self.input_summary, data_processor.sensor_summary], axis = 0).reset_index(drop=True)
                self.sensor_measurements = pd.merge(self.sensor_measurements, data_processor.sensor_measurements_pd, on='time', how = 'inner')
        data_processor.discard_data()
        self.data_processors.append(data_processor)
        
    

    def preprocess(self):
        """
        Generates train, validation, and test SHREDDataset objects.
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

        X_train_forecast, y_train_forecast = None, None
        X_valid_forecast, y_valid_forecast = None, None
        X_test_forecast, y_test_forecast = None, None

        # Process reconstructor datasets
        if self.reconstructor_flag:
            for dataset_dict in self.reconstructor_data_elements:
                X_train_recon, y_train_recon = concatenate_datasets(
                    X_train_recon, y_train_recon, dataset_dict['train'][0], dataset_dict['train'][1]
                )
                X_valid_recon, y_valid_recon = concatenate_datasets(
                    X_valid_recon, y_valid_recon, dataset_dict['validation'][0], dataset_dict['validation'][1]
                )
                X_test_recon, y_test_recon = concatenate_datasets(
                    X_test_recon, y_test_recon, dataset_dict['test'][0], dataset_dict['test'][1]
                )

        # Process forecastor datasets
        if self.forecastor_flag:
            for dataset_dict in self.forecastor_data_elements:
                X_train_forecast, y_train_forecast = concatenate_datasets(
                    X_train_forecast, y_train_forecast, dataset_dict['train'][0], dataset_dict['train'][1]
                )
                X_valid_forecast, y_valid_forecast = concatenate_datasets(
                    X_valid_forecast, y_valid_forecast, dataset_dict['validation'][0], dataset_dict['validation'][1]
                )
                X_test_forecast, y_test_forecast = concatenate_datasets(
                    X_test_forecast, y_test_forecast, dataset_dict['test'][0], dataset_dict['test'][1]
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

        X_train_forecast = torch.tensor(X_train_forecast, dtype=torch.float32, device=device)
        y_train_forecast = torch.tensor(y_train_forecast, dtype=torch.float32, device=device)
        X_valid_forecast = torch.tensor(X_valid_forecast, dtype=torch.float32, device=device)
        y_valid_forecast = torch.tensor(y_valid_forecast, dtype=torch.float32, device=device)
        X_test_forecast = torch.tensor(X_test_forecast, dtype=torch.float32, device=device)
        y_test_forecast = torch.tensor(y_test_forecast, dtype=torch.float32, device=device)

        # Create TimeSeriesDataset objects
        train_recon_dataset = TimeSeriesDataset(X_train_recon, y_train_recon)
        valid_recon_dataset = TimeSeriesDataset(X_valid_recon, y_valid_recon)
        test_recon_dataset = TimeSeriesDataset(X_test_recon, y_test_recon)

        train_forecast_dataset = TimeSeriesDataset(X_train_forecast, y_train_forecast)
        valid_forecast_dataset = TimeSeriesDataset(X_valid_forecast, y_valid_forecast)
        test_forecast_dataset = TimeSeriesDataset(X_test_forecast, y_test_forecast)

        SHRED_train_dataset = SHREDDataset(train_recon_dataset, train_forecast_dataset)
        SHRED_valid_dataset = SHREDDataset(valid_recon_dataset, valid_forecast_dataset)
        SHRED_test_dataset = SHREDDataset(test_recon_dataset, test_forecast_dataset)

        return SHRED_train_dataset, SHRED_valid_dataset, SHRED_test_dataset


    def postprocess(self, data, uncompress = None, unscale = None):
        uncompress = uncompress if uncompress is not None else self.compression is not None
        unscale = unscale if unscale is not None else self.scaling == "minmax" # prob change to boolean
        results = {}
        start_index = 0
        for data_processor in self.data_processors:
            field_spatial_dim = data_processor.Y_spatial_dim
            print('field_spatial_dim',field_spatial_dim)
            field_data = data[:, start_index:start_index+field_spatial_dim]
            if isinstance(data, torch.Tensor):
                field_data = field_data.cpu().numpy()
            start_index = field_spatial_dim + start_index
            print('field_data.shape',field_data.shape)
            field_data = data_processor.inverse_transform(field_data, uncompress, unscale)
            results[data_processor.id] = field_data
        return results


    def generate_X(self, start = None, end = None, measurements = None, time = None, forecaster=None):
        results = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start_sensor = 0
        if start is None and end is None:
            # generate lagged sequences from measurements
            for data_processor in self.data_processors:
                end_sensor = start_sensor + data_processor.sensor_measurements.shape[1]
                result = data_processor.transform_X(measurements[:,start_sensor:end_sensor])
                if results is None:
                    results = result
                else:
                    results = np.concatenate((results, result), axis = 1)
                start_sensor = end_sensor
            results = generate_lagged_sequences_from_sensor_measurements(results, self.lags)
        else:
            # generate lagged sequences from start to end (inclusive)
            for data_processor in self.data_processors:
                if data_processor.sensor_measurements is not None:
                    end_sensor = start_sensor + data_processor.sensor_measurements.shape[1]
                    if measurements is not None:
                        field_measurements = measurements[:,start_sensor:end_sensor]
                        result = data_processor.generate_X(end = end, measurements = field_measurements, time = time)
                    else:
                        result = data_processor.generate_X(end = end, measurements = None, time = time)
                    if results is None:
                        results = result
                    else:
                        results = np.concatenate((results, result), axis = 1)
                    start_sensor = end_sensor
            # extract indices without data (gaps)
            gap_indices = np.where(np.isnan(results).any(axis=1))[0]
            for gap in gap_indices:
                # gap is being forecasted, gap - 1 is 'current'
                gap_lagged_sequence = results[gap - 1 - self.lags:gap,:].copy()
                gap_lagged_sequence = gap_lagged_sequence[np.newaxis,:,:]
                gap_lagged_sequence = torch.tensor(gap_lagged_sequence, dtype=torch.float32, device=device)
                results[gap] = forecaster(gap_lagged_sequence).detach().numpy()
            results = generate_lagged_sequences_from_sensor_measurements(results, self.lags)
            results = results[start:end+1,:,:]
        results = torch.tensor(results, dtype=torch.float32, device=device)
        return results