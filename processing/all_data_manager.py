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

    # TODO: Recon, Predict, Forecast, will need sensor summary to notify user order of sensor measurements
    # TODO: sensor summary will also need to include parameter summary
    # TODO: each sensor and parameter can be identified to a dataset/field/regime with the unique id of the SHREDDataProcessor

    def __init__(self, parametric = False, lags = 20, time = None, train_size = 0.75, val_size = 0.15, test_size = 0.15,
                 scaling = True, compression = True, reconstructor=True, forecastor=True):
        self.parametric = parametric
        self.lags = lags # number of time steps to look back
        self.scaling = scaling
        self.compression = compression
        self.time = time
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.reconstructor_flag = reconstructor
        self.forecastor_flag = forecastor
    
        self.data_processors = [] # a list storing references to SHREDDataProcessor objects being managed
        self.reconstructor_indices = None # a dict storing 'train', 'validation', 'test' indices for SHRED reconstructor
        self.forecastor_indices = None # a dict storing 'train', 'validation', 'test' indices for SHRED forecastor

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.reconstructor_data_elements = []
        self.forecastor_data_elements = []
 
        # self.reconstructor = reconstructor # flag for generating datasets for SHRED reconstructor
        # self.forecastor = forecastor # flag for generating datasets for SHRED forecaster

    #TODO: allow for sensor measurments as well
    #TODO: allow for params as well, maybe a list of array, ea ch array shape (params,time)
    def add(self, data, random_sensors = None, stationary_sensors = None, mobile_sensors = None, params=None, compression=None, scaling=None, time=None, id=None, parametric = None):
        """
        Creates and adds a new SHREDDataProcessor object.
        - file path: file path or list of file paths (if multiple trajectories) to data (string, list)
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
        data = get_data(data)
        compression = compression if compression is not None else self.compression
        scaling = scaling if scaling is not None else self.scaling
        time = time if time is not None else self.time
        parametric = parametric if parametric is not None else self.parametric
        
        # generate train/val/test indices based on number of timesteps in initial SHREDDataProcessor object
        if len(self.data_processors) == 0:
            if parametric is False:
                # non-parametric case, train/val/test split by time indices
                self.reconstructor_indices = get_train_val_test_indices(len(time), self.train_size, self.val_size, self.test_size, method = "random")
                self.forecastor_indices = get_train_val_test_indices(len(time), self.train_size, self.val_size, self.test_size, method = "sequential")
            else:
                # parametric case, train/val/test split by parameters (trajectories)
                self.reconstructor_indices = get_train_val_test_indices(data.shape[0], self.train_size, self.val_size, self.test_size, method = "random")
                self.forecastor_indices = get_train_val_test_indices(data.shape[0], self.train_size, self.val_size, self.test_size, method = "sequential")
        
        # create and initialize SHREDData object
        data_processor = SHREDDataProcessor(
            parametric= self.parametric,
            params = params, # add method dependent, might add data from a field without params?
            data=data,
            random_sensors=random_sensors,
            stationary_sensors=stationary_sensors,
            mobile_sensors=mobile_sensors,
            lags=self.lags,
            time=time,
            compression=compression,
            scaling=scaling,
            id=id,
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