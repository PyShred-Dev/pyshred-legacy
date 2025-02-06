


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

    METHODS = {
        'all': ['random_reconstructor', 'temporal_reconstructor', 'sensor_forecaster'],
        'reconstruct': ['random_reconstructor'],
        'predict': ['temporal_reconstructor'],
        'forecast': ['sensor_forecaster', 'temporal_reconstructor']
    }


    def __init__(self, lags = 20, time = None, train_size = 0.75, val_size = 0.15, test_size = 0.15, scaling = True, compression = True, method = 'all'):
        self.scaling = scaling
        self.compression = compression
        self.time = time # expects a 1D numpy array
        self.lags = lags # number of time steps to look back
        self.data_processors = [] # a list storing SHREDDataProcessor objects
        self.random_indices = None # a dict storing 'train', 'validation', 'test' indices for SHRED reconstructor
        self.sequential_indices = None # a dict storing 'train', 'validation', 'test' indices for SHRED forecastor
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.random_reconstructor = []
        self.sensor_forecaster = []
        self.temporal_reconstructor = []
        # self.reconstructor_flag = reconstructor
        # self.forecastor_flag = forecastor
        self.input_summary = None #
        self.sensor_measurements = None # all sensor measurement
        self.method = method
        # self.reconstructor = reconstructor # flag for generating datasets for SHRED reconstructor
        # self.forecastor = forecastor # flag for generating datasets for SHRED forecaster

    def add_field(self, data, random_sensors = None, stationary_sensors = None, mobile_sensors = None, compression = None, id = None, scaling = None, time = None):
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

        # save sensor-related information to sensor-related attributes
        if data_processor.sensor_summary is not None and data_processor.sensor_measurements_pd is not None:
            if self.input_summary is None and self.sensor_measurements is None:
                self.input_summary = data_processor.sensor_summary
                self.sensor_measurements = data_processor.sensor_measurements_pd
            else:
                self.input_summary = pd.concat([self.input_summary, data_processor.sensor_summary], axis = 0).reset_index(drop=True)
                self.sensor_measurements = pd.merge(self.sensor_measurements, data_processor.sensor_measurements_pd, on='time', how = 'inner')


        # generate train/val/test indices
        if len(self.data_processors) == 0:
            self.random_indices = get_train_val_test_indices(len(time), self.train_size, self.val_size, self.test_size, method = "random")
            self.sequential_indices = get_train_val_test_indices(len(time), self.train_size, self.val_size, self.test_size, method = "sequential")
            print('self.random_indices["train"]', self.random_indices["train"])


        if 'random_reconstructor' in self.METHODS[self.method]:
            dataset_dict = data_processor.generate_dataset(
                self.random_indices['train'],
                self.random_indices['validation'],
                self.random_indices['test'],
                method='random_reconstructor'
            )
            self.random_reconstructor.append(dataset_dict)

        if 'temporal_reconstructor' in self.METHODS[self.method]:
            dataset_dict = data_processor.generate_dataset(
                self.sequential_indices['train'],
                self.sequential_indices['validation'],
                self.sequential_indices['test'],
                method='temporal_reconstructor' # different name so different scalers etc.
            )
            self.temporal_reconstructor.append(dataset_dict)

        if 'sensor_forecaster' in self.METHODS[self.method]:
            dataset_dict = data_processor.generate_dataset(
                self.sequential_indices['train'],
                self.sequential_indices['validation'],
                self.sequential_indices['test'],
                method='sensor_forecaster'
            )
            self.sensor_forecaster.append(dataset_dict)

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

        # Initialize datasets
        X_train_random_reconstructor, y_train_random_reconstructor = None, None
        X_valid_random_reconstructor, y_valid_random_reconstructor = None, None
        X_test_random_reconstructor, y_test_random_reconstructor = None, None

        X_train_temporal_reconstructor, y_train_temporal_reconstructor = None, None
        X_valid_temporal_reconstructor, y_valid_temporal_reconstructor = None, None
        X_test_temporal_reconstructor, y_test_temporal_reconstructor = None, None

        X_train_sensor_forecaster, y_train_sensor_forecaster = None, None
        X_valid_sensor_forecaster, y_valid_sensor_forecaster = None, None
        X_test_sensor_forecaster, y_test_sensor_forecaster = None, None

        # Process random reconstructor datasets
        if 'random_reconstructor' in self.METHODS[self.method]:
            for dataset_dict in self.random_reconstructor:
                X_train_random_reconstructor, y_train_random_reconstructor = concatenate_datasets(
                    X_train_random_reconstructor, y_train_random_reconstructor, dataset_dict['train'][0], dataset_dict['train'][1]
                )
                X_valid_random_reconstructor, y_valid_random_reconstructor = concatenate_datasets(
                    X_valid_random_reconstructor, y_valid_random_reconstructor, dataset_dict['validation'][0], dataset_dict['validation'][1]
                )
                X_test_random_reconstructor, y_test_random_reconstructor = concatenate_datasets(
                    X_test_random_reconstructor, y_test_random_reconstructor, dataset_dict['test'][0], dataset_dict['test'][1]
                )

        # Process random reconstructor datasets
        if 'temporal_reconstructor' in self.METHODS[self.method]:
            for dataset_dict in self.temporal_reconstructor:
                X_train_temporal_reconstructor, y_train_temporal_reconstructor = concatenate_datasets(
                    X_train_temporal_reconstructor, y_train_temporal_reconstructor, dataset_dict['train'][0], dataset_dict['train'][1]
                )
                X_valid_temporal_reconstructor, y_valid_temporal_reconstructor = concatenate_datasets(
                    X_valid_temporal_reconstructor, y_valid_temporal_reconstructor, dataset_dict['validation'][0], dataset_dict['validation'][1]
                )
                X_test_temporal_reconstructor, y_test_temporal_reconstructor = concatenate_datasets(
                    X_test_temporal_reconstructor, y_test_temporal_reconstructor, dataset_dict['test'][0], dataset_dict['test'][1]
                )


        # Process forecastor datasets
        if 'sensor_forecaster' in self.METHODS[self.method]:
            for dataset_dict in self.sensor_forecaster:
                X_train_sensor_forecaster, y_train_sensor_forecaster = concatenate_datasets(
                    X_train_sensor_forecaster, y_train_sensor_forecaster, dataset_dict['train'][0], dataset_dict['train'][1]
                )
                X_valid_sensor_forecaster, y_valid_sensor_forecaster = concatenate_datasets(
                    X_valid_sensor_forecaster, y_valid_sensor_forecaster, dataset_dict['validation'][0], dataset_dict['validation'][1]
                )
                X_test_sensor_forecaster, y_test_sensor_forecaster = concatenate_datasets(
                    X_test_sensor_forecaster, y_test_sensor_forecaster, dataset_dict['test'][0], dataset_dict['test'][1]
                )


        # Create TimeSeriesDataset and SHREDDataset objects
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert data to torch tensors and move to the specified device
        X_train_random_reconstructor = torch.tensor(X_train_random_reconstructor, dtype=torch.float32, device=device)
        y_train_random_reconstructor = torch.tensor(y_train_random_reconstructor, dtype=torch.float32, device=device)
        X_valid_random_reconstructor = torch.tensor(X_valid_random_reconstructor, dtype=torch.float32, device=device)
        y_valid_random_reconstructor = torch.tensor(y_valid_random_reconstructor, dtype=torch.float32, device=device)
        X_test_random_reconstructor = torch.tensor(X_test_random_reconstructor, dtype=torch.float32, device=device)
        y_test_random_reconstructor = torch.tensor(y_test_random_reconstructor, dtype=torch.float32, device=device)

        X_train_temporal_reconstructor = torch.tensor(X_train_temporal_reconstructor, dtype=torch.float32, device=device)
        y_train_temporal_reconstructor = torch.tensor(y_train_temporal_reconstructor, dtype=torch.float32, device=device)
        X_valid_temporal_reconstructor = torch.tensor(X_valid_temporal_reconstructor, dtype=torch.float32, device=device)
        y_valid_temporal_reconstructor = torch.tensor(y_valid_temporal_reconstructor, dtype=torch.float32, device=device)
        X_test_temporal_reconstructor = torch.tensor(X_test_temporal_reconstructor, dtype=torch.float32, device=device)
        y_test_temporal_reconstructor = torch.tensor(y_test_temporal_reconstructor, dtype=torch.float32, device=device)

        X_train_sensor_forecaster = torch.tensor(X_train_sensor_forecaster, dtype=torch.float32, device=device)
        y_train_sensor_forecaster = torch.tensor(y_train_sensor_forecaster, dtype=torch.float32, device=device)
        X_valid_sensor_forecaster = torch.tensor(X_valid_sensor_forecaster, dtype=torch.float32, device=device)
        y_valid_sensor_forecaster = torch.tensor(y_valid_sensor_forecaster, dtype=torch.float32, device=device)
        X_test_sensor_forecaster = torch.tensor(X_test_sensor_forecaster, dtype=torch.float32, device=device)
        y_test_sensor_forecaster = torch.tensor(y_test_sensor_forecaster, dtype=torch.float32, device=device)

        # Create TimeSeriesDataset objects
        train_random_reconstructor_dataset = TimeSeriesDataset(X_train_random_reconstructor, y_train_random_reconstructor)
        valid_random_reconstructor_dataset = TimeSeriesDataset(X_valid_random_reconstructor, y_valid_random_reconstructor)
        test_random_reconstructor_dataset = TimeSeriesDataset(X_test_random_reconstructor, y_test_random_reconstructor)

        train_temporal_reconstructor_dataset = TimeSeriesDataset(X_train_temporal_reconstructor, y_train_temporal_reconstructor)
        valid_temporal_reconstructor_dataset = TimeSeriesDataset(X_valid_temporal_reconstructor, y_valid_temporal_reconstructor)
        test_temporal_reconstructor_dataset = TimeSeriesDataset(X_test_temporal_reconstructor, y_test_temporal_reconstructor)


        train_sensor_forecaster_dataset = TimeSeriesDataset(X_train_sensor_forecaster, y_train_sensor_forecaster)
        valid_sensor_forecaster_dataset = TimeSeriesDataset(X_valid_sensor_forecaster, y_valid_sensor_forecaster)
        test_sensor_forecaster_dataset = TimeSeriesDataset(X_test_sensor_forecaster, y_test_sensor_forecaster)

        SHRED_train_dataset = SHREDDataset(train_random_reconstructor_dataset, train_temporal_reconstructor_dataset, train_sensor_forecaster_dataset)
        SHRED_valid_dataset = SHREDDataset(valid_random_reconstructor_dataset, valid_temporal_reconstructor_dataset, valid_sensor_forecaster_dataset)
        SHRED_test_dataset = SHREDDataset(test_random_reconstructor_dataset, test_temporal_reconstructor_dataset, test_sensor_forecaster_dataset)

        return SHRED_train_dataset, SHRED_valid_dataset, SHRED_test_dataset


    def postprocess(self, data, uncompress = True, unscale = True, method = None):
        # uncompress = uncompress if uncompress is not None else self.compression is not None
        # unscale = unscale if unscale is not None else self.scaling == "minmax" # prob change to boolean
        results = {}
        start_index = 0
        for data_processor in self.data_processors:
            field_spatial_dim = data_processor.Y_spatial_dim
            # print('field_spatial_dim',field_spatial_dim)
            field_data = data[:, start_index:start_index+field_spatial_dim]
            if isinstance(data, torch.Tensor):
                field_data = field_data.cpu().numpy()
            start_index = field_spatial_dim + start_index
            # print('field_data.shape',field_data.shape)
            field_data = data_processor.inverse_transform(field_data, uncompress, unscale, method)
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