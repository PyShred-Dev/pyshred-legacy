import copy
from ...processing.utils import generate_lagged_sequences_from_sensor_measurements, l2
from ..decoder_models import *
from ..sequence_models import *
from .sensor_forecaster import SENSOR_FORECASTER
from .reconstructor import RECONSTRUCTOR
from ..decoder_models.abstract_decoder import AbstractDecoder
from ..sequence_models.abstract_sequence import AbstractSequence
import torch

# model registry
SEQUENCE_MODELS = {
    "LSTM": LSTM,
}

DECODER_MODELS = {
    "SDN": SDN,
}


class SHRED():
    """
    SHallow REcurrent Decoder (SHRED) neural network architecture. SHRED learns a mapping from
    trajectories of sensor measurements to high-dimensional, spatio-temporal states.

    Parameters:
    -----------
    sequence : {"LSTM"}, default="LSTM"
        The sequence model used in SHRED.
    decoder : {"SDN"}, default="SDN"
        The decoder model used in SHRED.

    Attributes:
    -----------
    sensor_summary : pandas.DataFrame
        Summary of the sensors seen during `fit`
        - row index: row sensor belongs to in `sensor_measurements`
        - dataset: dataset the sensor belong to
        - type: either stationary (immobile) or mobile
        - location/trajectory: 
            - if `type` is stationary: location of sensor represented as tuple
            - if `type` is mobile: trajectory of sensor represented as a list of tuples 

    sensor_data : 2d-array of shape (n_sensors, n_timesteps)
        Sensor data used during `fit`
    
    reconstructor_val_errors : numpy.ndarray
        History of reconstructor val errors at each training epoch.
    
    recon_forecast_val_errors : numpy.ndarray
        History of sensor_forecaster val errors at each training epoch.

    Methods:
    --------
    fit(data, sensors, lags = 40, time = None, sensor_forecaster = True, n_components = 20, val_size = 0.2, batch_size=64, num_epochs=4000, lr=1e-3, verbose=True, patience=20):
        Fit SHRED on trajectories of sensor measurements to perform reconstructions and forecasts of high-dimensional state spaces.
    
    summary():
        Prints out a summary of the fitted SHRED model.
    
    predict(start, end, sensor_data = None, sensor_data_time = None):
        Takes in a start and end time (required). Optional sensor_data and sensor_data_time can be
        added to improve forecasts (out-of-sample reconstructions).
    
    recon(sensor_measurments):
        Performs full-state reconstructin using only the provided sensor_measurements.

    forecast(timesteps, sensor_data = None, sensor_data_time = None):
        Forecast the high-dimensional state space `timesteps` into the future.
        It is a convnience wrapper around `predict(self, start)` for forecasts (out-of-sample reconstructions).

    Notes:
    ------
    * The `fit` method requires the number of timesteps in `data` to exceed the value of `lags` (default = 40).
    * `lags` represents the number of timesteps the model looks back for.
    * By default SHRED trains on the first 20 principal components of the dataset. For maximum resolution reconstructions,
      skip dimensionality reduction by setting `n_components=None` when calling `fit()`.

    References:
    -----------
    [1] Jan P. Williams, Olivia Zahn, and J. Nathan Kutz, "Sensing with shallow recurrent
        decoder networks", arXiv:2301.12011, 2024. Available: https://arxiv.org/abs/2301.12011
    
    [2] M.R. Ebers, J.P. Williams, K.M. Steele, and J.N. Kutz, "Leveraging Arbitrary Mobile
        Sensor Trajectories With Shallow Recurrent Decoder Networks for Full-State Reconstruction,"
        IEEE Access, vol. 12, pp. 97428-97439, 2024. doi: 10.1109/ACCESS.2024.3423679.
    
    [3] J. Nathan Kutz, Maryam Reza, Farbod Faraji, and Aaron Knoll, "Shallow Recurrent Decoder
        for Reduced Order Modeling of Plasma Dynamics", arXiv:2405.11955, 2024. Available: https://arxiv.org/abs/2405.11955
    
    Examples:
    ---------
    >>> from pyshred import SHRED
    >>> shred = SHRED(sequence = 'LSTM', decoder='SDN')
    >>> shred.fit(data = data, sensors = sensor, n_components = 20, sensor_forecaster = True)
    ---------

    """


    def __init__(self, sequence = 'LSTM', decoder = 'SDN'):
        """
        Initialize SHRED with a sequence model and a decoder model.
        
        Parameters:
        ----------
        sequence : str or AbstractSequence
            The sequence model to use. Either a string ('LSTM', etc.) or an instance of a sequence model.
        decoder : str or AbstractDecoder
            The decoder model to use. Either a string ('SDN', etc.) or an instance of a decoder model.
        """
        super().__init__()
        # Initialize Sequence Model
        if isinstance(sequence, AbstractSequence):
            self._sequence_model_reconstructor = copy.deepcopy(sequence)
            self._sequence_model_predictor = copy.deepcopy(sequence)
            self._sequence_model_sensor_forecaster = copy.deepcopy(sequence)
        elif isinstance(sequence, str):
            sequence = sequence.upper()
            if sequence not in SEQUENCE_MODELS:
                raise ValueError(f"invalid sequence model: {sequence}. Choose from: {list(SEQUENCE_MODELS.keys())}")
            self._sequence_model_reconstructor = SEQUENCE_MODELS[sequence]()
            self._sequence_model_predictor = SEQUENCE_MODELS[sequence]()
            self._sequence_model_sensor_forecaster = SEQUENCE_MODELS[sequence]()
        else:
            raise ValueError("invalid type for 'sequence'. Must be str or an AbstractSequence instance.")

        # Initialize Decoder Model
        if isinstance(decoder, AbstractDecoder):
            self._decoder_model_reconstructor = copy.deepcopy(decoder)
            self._decoder_model_predictor = copy.deepcopy(decoder)
            self._decoder_model_sensor_forecaster = copy.deepcopy(decoder)

        elif isinstance(decoder, str):
            decoder = decoder.upper()
            if decoder not in DECODER_MODELS:
                raise ValueError(f"invalid decoder model: {decoder}. Choose from: {list(DECODER_MODELS.keys())}")
            self._decoder_model_reconstructor = DECODER_MODELS[decoder]()
            self._decoder_model_predictor = DECODER_MODELS[decoder]()
            self._decoder_model_sensor_forecaster = DECODER_MODELS[decoder]()
        else:
            raise ValueError("invalid type for 'decoder'. Must be str or an AbstractDecoder instance.")

        self.sensor_forecaster = None
        self.reconstructor = None
        self.predictor = None

        self.reconstructor_val_errors = None
        self.predictor_val_errors = None
        self.sensor_forecaster_val_errors = None

    def fit(self, train_dataset, val_dataset,  batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=50):
        """
        Train SHRED using the high-dimensional state space data.

        Parameters:
        -----------
        data : dict, required
            A dictionary of high-dimensional state space datasets where:
            - The key(s) are strings used as dataset identifiers
            - The values are numpy arrays representating the datasets.
            The last dimension of each array must represent the number of timesteps.
        
        sensors : dict, required
            A dictionary of sensor locations where: 
            - The key(s) are strings used as dataset identifiers
            - The values can be one of the following:
                - **int**: The number of sensors. Sensors will be randomly placed.
                - **list of tuples**: For stationary sensors, where each tuple represents the
                        location of a sensor in the spatial domain.
                - **list of lists of tuples**: For mobile sensors, where each list represents a
                        sensors location over time. Each inner list contains tuples for each timestep,
                        corresponding to the sensor's location at that timestep.
        
        lags : int, optional
            The number of timesteps the sequence model looks back at.

        time : numpy array, optional
            A 1D numpy array of evenly spaced, strictly increasing timestamps. Each element corresponds to a timestep
            in the last dimension of the data arrays.
            Default is a 1D numpy array ranging from 0 to `N-1`, where `N` is the size of the
            last dimension of the dataset.

        sensor_forecaster : bool, optional
            If True, trains a sensor forecaster for performing out-of-sample reconstructions.
            Default is True.

        n_components : int/None, optional
            A integer representing the number of principal components to keep after rSVD.
            If None, no compression will be performed. Default is 20.
        
        val_size : float, optional
            A float representing the proportion of the dataset to allocate for val.
            Default is 0.2.
        
        batch_size : int, optional
            Number of samples per batch for training. Default is 64.

        num_epochs : int, optional
            Number of epochs for training the model. Default is 4000.

        lr : float, optional
            Learning rate for the optimizer. Default is 1e-3.
        
        verbose : bool, optional
            If True, prints progress during training. Default is True.

        patience : int, optional
            Number of epochs to wait for improvement before early stopping. Default is 20.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ########################################### SHRED Reconstructor #################################################
        if hasattr(train_dataset, "reconstructor_dataset"):
            train_set = train_dataset.reconstructor_dataset
            val_set = val_dataset.reconstructor_dataset
            input_size = train_set.X.shape[2] # nsensors + nparams
            output_size = val_set.Y.shape[1]
            self._sequence_model_reconstructor.initialize(input_size) # initialize with nsensors
            self._sequence_model_reconstructor.to(device)
            self._decoder_model_reconstructor.initialize(input_size = self._sequence_model_reconstructor.output_size, output_size=output_size) # could pass in entire sequence model
            self._decoder_model_reconstructor.to(device)
            self.reconstructor = RECONSTRUCTOR(sequence=self._sequence_model_reconstructor,
                                                      decoder=self._decoder_model_reconstructor).to(device)
            print("\nFitting Reconstructor...")
            self.reconstructor_val_errors = self.reconstructor.fit(model = self.reconstructor, train_dataset = train_set, val_dataset = val_set,
                                                                num_sensors = input_size, output_size = output_size
                                    , batch_size = batch_size, num_epochs = num_epochs, lr = lr, verbose = verbose, patience = patience)
        

        ########################################### SHRED Predictor #################################################
        if hasattr(train_dataset, "predictor_dataset"):
            train_set = train_dataset.predictor_dataset
            val_set = val_dataset.predictor_dataset
            input_size = train_set.X.shape[2] # nsensors + nparams
            output_size = val_set.Y.shape[1]
            self._sequence_model_predictor.initialize(input_size) # initialize with nsensors
            self._sequence_model_predictor.to(device)
            self._decoder_model_predictor.initialize(input_size = self._sequence_model_predictor.output_size, output_size=output_size) # could pass in entire sequence model
            self._decoder_model_predictor.to(device)
            self.predictor = RECONSTRUCTOR(sequence=self._sequence_model_predictor,
                                                        decoder=self._decoder_model_predictor).to(device)
            print("\nFitting Predictor...")
            self.predictor_val_errors = self.predictor.fit(model = self.predictor, train_dataset = train_set, val_dataset = val_set,
                                                                num_sensors = input_size, output_size = output_size
                                    , batch_size = batch_size, num_epochs = num_epochs, lr = lr, verbose = verbose, patience = patience)
        
        
        ########################################### SHRED Sensor Forecaster ####################################################
        if hasattr(train_dataset, "sensor_forecaster_dataset"):
            train_set = train_dataset.sensor_forecaster_dataset
            val_set = val_dataset.sensor_forecaster_dataset
            input_size = train_set.X.shape[2] # nsensors + nparams
            output_size = val_set.Y.shape[1]
            self._sequence_model_sensor_forecaster.initialize(input_size)
            self._sequence_model_sensor_forecaster.to(device)
            self._decoder_model_sensor_forecaster.initialize(self._sequence_model_sensor_forecaster.output_size, output_size)
            self._decoder_model_sensor_forecaster.to(device)
            self.sensor_forecaster = SENSOR_FORECASTER(sequence=self._sequence_model_sensor_forecaster,
                                                decoder=self._decoder_model_sensor_forecaster).to(device)
            # self.sensor_forecaster = _SHRED_FORECASTER(model=LSTM without decoder)
            print("\nFitting Sensor Forecaster...")
            self.sensor_forecaster_val_errors =  self.sensor_forecaster.fit(model = self.sensor_forecaster, train_dataset = train_set, val_dataset = val_set, num_sensors = input_size,
                                        output_size = output_size, batch_size = batch_size, num_epochs = num_epochs,
                                        lr = lr, verbose = verbose, patience = patience)
            
        result = {}
        if self.reconstructor_val_errors is not None:
            result['reconstruction_val_errors'] = self.reconstructor_val_errors
        if self.predictor_val_errors is not None:
            result['prediction_val_errors'] = self.predictor_val_errors
        if self.sensor_forecaster_val_errors is not None:
            result['sensor_forecast_val_errors'] = self.sensor_forecaster_val_errors

        return result

    def predict(self, x):
        return self.predictor(x).detach().cpu().numpy()

    def reconstruct(self, x):
        return self.reconstructor(x).detach().cpu().numpy()


    








    


    




        
        # test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
        # test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
        # print('Test Reconstruction Error: ')
        # print(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))

    # def predict(self, start, end, sensor_data = None, sensor_data_time = None):
    #     """
    #     Takes in a start and end time (required). Optional sensor_data and sensor_data_time can be
    #     added to improve forecasts (out-of-sample reconstructions).
    #     """
    #     # Check if fit() method has been called prior
    #     if not self._is_fitted:
    #         raise RuntimeError("The SHRED model must be fit before calling recon().")
    #     ########################################## VALIDATE USER INPUT ############################################
    #     time_step = self._time[1] - self._time[0]
    #     if not isinstance(start, (int, np.integer)):
    #         raise TypeError(f"Expected 'start' to be an integer, but got {type(start).__name__}.")
    #     if not isinstance(end, (int, np.integer)):
    #         raise TypeError(f"Expected 'end' to be an integer, but got {type(start).__name__}.")
    #     start_time = start # inclusive start time
    #     end_time = end # inclusive end time
    #     # Check if start time less than or equal to end time
    #     if start_time > end_time:
    #         raise ValueError(f"Start time ({start_time}) must be less than or equal to end time ({end_time}).")
    #     # Check if start time is greater than train data start time + lag time
    #     if start_time < self._time[0] + (self._lags * time_step):
    #         raise ValueError(f"Start time must be greater or equal to {self._time[0] + (self._lags * time_step)}")
    #     # Check if start time is val 
    #     if (start_time - self._time[0])%time_step != 0:
    #         raise ValueError(f"Start time ({start_time}) is invalid.")
    #     # Check if end time is val 
    #     if (end_time - self._time[0])%time_step != 0:
    #         raise ValueError(f"End time ({end_time}) is invalid.")
    #     if sensor_data is not None:
    #         # Check if time exists
    #         if sensor_data_time is None:
    #             raise ValueError("The 'sensor_data_time' corresponding to 'sensor_data' does not exist.")
    #         # Check if sensor_data same timesteps as time
    #         if sensor_data.shape[1] != sensor_data_time.shape[0]:
    #             raise ValueError(f"The number of columns in 'sensor_data' ({sensor_data.shape[1]}) must match length of 'time' ({sensor_data_time.shape[0]}).")
    #         # Check for expected number of sensors in sensor_data
    #         if sensor_data.shape[0] != self.sensor_data.shape[0]:
    #             raise ValueError(f"Expected {self.sensor_data.shape[0]} sensors but got {sensor_data.shape[0]} in 'sensor_data'.")
    #         if np.any(sensor_data_time % time_step != 0):
    #             raise ValueError(f"All values in 'time' must be multiples of {time_step}")
    #         if np.any(sensor_data_time <= self._time[-1]):
    #             print(f"Warning: Some values in 'time' are less than {self._time[-1]}. Any 'sensor_data' value with a corresponding 'time' value less than {self._time[-1]} will be ignored.")
    #         # Scale input sensor data
    #         scaled_sensor_data_in = self._scale_sensor_data(sensor_data)
    #     if start_time <= self._time[-1]:
    #         start_time_index = np.where(self._time == start_time)[0][0]
    #     else:
    #         start_time_index = (len(self._time) - 1) + int((start_time - self._time[-1])/time_step)
    #     end_time_index = int((end_time - start_time) / time_step) + start_time_index
    #     scaled_sensor_data = self._scale_sensor_data(self.sensor_data)
    #     if end_time_index < len(self._time): # If we don't need forecast at all (ignore argument time and sensor_data)
    #         sensor_measurements_scaled = scaled_sensor_data[:,start_time_index - self._lags : end_time_index + 1] # +1 to be inclusive of endpoint, timesteps as columns
    #     else: # Forecasting is necessary
    #         if start_time_index < len(self._time):
    #             sensor_measurements_scaled = scaled_sensor_data[:,start_time_index - self._lags:].T
    #         else:
    #             sensor_measurements_scaled = scaled_sensor_data[:,-self._lags:].T # get last lag_index number of timesteps, timesteps represented by rows
    #         n_forecasts = end_time_index - (len(self._time) - 1)
    #         initial_in = sensor_measurements_scaled
    #         device = 'cuda' if next(self.sensor_forecaster.parameters()).is_cuda else 'cpu'
    #         initial_in = torch.tensor(initial_in, dtype=torch.float32).to(device).unsqueeze(0) # add a dimension 
    #         vals = []
    #         # append initial sensor data (not forecasted sensor data) to vals
    #         for i in range(0, initial_in[0].shape[0]):
    #             vals.append(initial_in[0, i,:].detach().cpu().clone().numpy())
    #         num_sensors = self.sensor_data.shape[0]
    #         time_index_list = np.array([]) # initialize time_index_list
    #         for i in range(n_forecasts):
    #             if sensor_data is not None:
    #                 i_time = self._time[-1] + (i+1)*time_step
    #                 time_index_list = np.where(sensor_data_time == i_time)[0]
    #             if time_index_list.size > 0: # i_time exists in 'time'
    #                 time_index = time_index_list[0]
    #                 scaled_sensor_forecast = scaled_sensor_data_in[:,time_index]
    #             else: # i_time does not exist in 'time'
    #                 scaled_sensor_forecast = self.sensor_forecaster(initial_in).detach().cpu().numpy()
    #             vals.append(scaled_sensor_forecast.reshape(num_sensors))
    #             temp = initial_in.clone()
    #             initial_in[0,:-1] = temp[0,1:]
    #             initial_in[0,-1] = torch.tensor(scaled_sensor_forecast)
    #         device = 'cuda' if next(self.reconstructor.parameters()).is_cuda else 'cpu'
    #         sensor_measurements_scaled_all = np.array(vals).T # timesteps as columns
    #         sensor_measurements_scaled = sensor_measurements_scaled_all[:,-(end_time_index - start_time_index) - self._lags - 1:]
    #     sensor_measurements_unscaled_recon = self._unscale_sensor_data(sensor_measurements_scaled)
    #     # Get reconstructions
    #     recon_dict = self.recon(sensor_measurments = sensor_measurements_unscaled_recon)
    #     return ReconstructionResult(recon_dict=recon_dict, sensor_measurements=sensor_measurements_unscaled_recon, time= np.arange(start_time, end_time + time_step, time_step))

    # def recon(self, sensor_measurments):
    #     """
    #     Performs full-state reconstructin using only the provided sensor_measurements.

    #     Parameters:
    #     -----------
    #     sensor_measurments : numpy array
    #         A numpy array sensor measurements where:
    #         - rows represent sensors (see sensor column order with .sensor_summary)
    #         - columns represents timesteps, the number of timesteps must be greater than `lag_index`.
    #         - ATTENTION: only seansor measurements past the first `lag_index` number of timesteps
    #         be reconstructed
    #     """
    #     sensor_measurments_scaled = self._scale_sensor_data(sensor_measurments).T # timesteps as rows
    #     n = sensor_measurments_scaled.shape[0]
    #     num_sensors = sensor_measurments_scaled.shape[1] # VALIDATE data in using self.num_sensors?
    #     data_in = np.zeros((n - self._lags, self._lags, num_sensors))
    #     for i in range(len(data_in)):
    #         data_in[i] = sensor_measurments_scaled[i:i+self._lags]
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     data_in = torch.tensor(data_in, dtype = torch.float32).to(device)
    #     with torch.no_grad():
    #         recon = self.reconstructor(data_in)
    #     recon_sensor_data_scaled = recon.detach().cpu().numpy()[:,:num_sensors] # timesteps as rows
    #     recon_fullstate_data_scaled = recon.detach().cpu().numpy()[:,num_sensors:] # timesteps as rows
    #     recon_sensor_data = np.empty_like(recon_sensor_data_scaled.T)
    #     for dataset_name, sc in self._sc_sensor_dict.items():
    #         indices = self.sensor_summary[self.sensor_summary['dataset'] == dataset_name]['row index']
    #         recon_sensor_data[indices] = sc.inverse_transform(recon_sensor_data_scaled[:,indices]).T # time as columns
    #     keys = ['sensors'] + self._data_keys
    #     recon_dict = {key: None for key in keys}
    #     start_index = 0
    #     for key in recon_dict:
    #         if key == 'sensors':
    #             recon_dict[key] = recon_sensor_data
    #         elif self._compressed:
    #             u = self._u_dict[key]
    #             s = self._s_dict[key]
    #             v_scaled = recon_fullstate_data_scaled[:, start_index: start_index + s.shape[0]] # s.shape[0] = number of components
    #             v = self._sc_data_dict[key].inverse_transform(v_scaled)
    #             svd_recon_flat = (u @ np.diag(s) @ v.T).T # timesteps is represented by rows
    #             recon_dict[key] = unflatten(data = svd_recon_flat, spatial_shape=self._data_spatial_shape[key])
    #             start_index += s.shape[0]
    #         else:
    #             ### Compression skipped during fit:
    #             ### num rows = timesteps
    #             ### num columns = num sensors + spatial_flattened (X1) + ... + spatial_flaxxened (Xn)
    #             recon_fullstate_data = self._sc_data_dict[key].inverse_transform(recon_fullstate_data_scaled[:, start_index:start_index + np.prod(self._data_spatial_shape[key])])
    #             recon_dict[key] = unflatten(data = recon_fullstate_data, spatial_shape = self._data_spatial_shape[key])
    #             start_index += np.prod(self._data_spatial_shape[key])
    #     return recon_dict

    # def forecast(self, timesteps, sensor_data = None, sensor_data_time = None):
    #     """
    #     Forecast the high-dimensional state space `timesteps` into the future.
    #     It is a convnience wrapper around `predict(self, start)` for forecasts (out-of-sample reconstructions).
    #     """
    #     if not self._is_fitted:
    #         raise RuntimeError("The SHRED model must be fit before calling forecast().")
    #     time_step = self._time[1] - self._time[0]
    #     start = self._time[-1] + time_step # first out-of-sample time
    #     end = start + (timesteps-1) * time_step # minus one since start is first out-of-sample time
    #     return self.predict(start = start, end = end, sensor_data = sensor_data, sensor_data_time = sensor_data_time)
    
    # def summary(self):
    #     """
    #     Prints out a summary of the fitted SHRED model.
    #     """
    #     if not self._is_fitted:
    #         raise RuntimeError("The SHRED model must be fit before calling summary().")
    #     total_width = 60
    #     between_width = 30
    #     summary = (
    #         f"{'SHRED Model Results':^60}\n"
    #         f"{'='*total_width}\n"
    #     )
    #     summary += f"{'Reconstructor':^60}\n"
    #     f"{'-'*total_width}\n"
    #     summary +=f"{'Sequence:':<{between_width}}{self.reconstructor._sequence_str}\n"
    #     summary += f"{'Decoder:':<{between_width}}{self.reconstructor._decoder_str}\n"
    #     summary += f"{'val Error (L2):':<{between_width}}{self.reconstructor._best_L2_error:.3f}\n"
    #     if self.sensor_forecaster is not None:
    #         summary += f"{'-'*total_width}\n"
    #         summary += f"{'Sensor Forecaster':^60}\n"
    #         f"{'-'*total_width}\n"
    #         summary += f"{'Sequence:':<{between_width}}{self.sensor_forecaster._sequence_str}\n"
    #         summary += f"{'Decoder:':<{between_width}}{self.sensor_forecaster._decoder_str}\n"
    #         summary += f"{'val Error (L2):':<{between_width}}{self.sensor_forecaster._best_L2_error:.3f}\n"
    #     summary += f"{'='*total_width}\n"
    #     summary += f"{'No. Observations:':<{between_width}}{len(self._time)}\n"
    #     summary += f"{'No. Sensors:':<{between_width}}{self.sensor_data.shape[0]}\n"
    #     summary += f"{'Time:':<{between_width}}(start: {self._time[0]}, end: {self._time[-1]}, by: {self._time[1] - self._time[0]})\n"
    #     summary += f"{'Lags (timesteps):':<{between_width}}{self._lags}\n"
    
    #     # Check for compression and append the appropriate string
    #     if self._compressed:
    #         summary += f"{'Compression (components):':<{between_width}}{self._n_components}\n"
    #     else:
    #         summary += f"{'Compression:':<{between_width}}{self._compressed}\n"
    #     summary += f"{'='*total_width}\n"
    #     print(summary)

    # # Takes in unscaled sensor data with time as columns
    # # Returns scaled sensor data with time as columns
    # def _scale_sensor_data(self, unscaled_sensor_data):
    #     scaled_sensor_data = np.empty_like(unscaled_sensor_data)
    #     for dataset_name, sc in self._sc_sensor_dict.items():
    #         indices = self.sensor_summary[self.sensor_summary['dataset'] == dataset_name]['row index']
    #         scaled_sensor_data[indices] = sc.transform(unscaled_sensor_data[indices].T).T
    #     return scaled_sensor_data
    
    # # Takes in scaled sensor data with time as columns
    # # Returns unscaled sensor data with time as columns
    # def _unscale_sensor_data(self, scaled_sensor_data):
    #     unscaled_sensor_data = np.empty_like(scaled_sensor_data)
    #     for dataset_name, sc in self._sc_sensor_dict.items():
    #         indices = self.sensor_summary[self.sensor_summary['dataset'] == dataset_name]['row index']
    #         unscaled_sensor_data[indices] = sc.inverse_transform(scaled_sensor_data[indices].T).T
    #     return unscaled_sensor_data


# # class _SHRED_RECONSTRUCTOR(nn.Module):
# class _SHRED(nn.Module):
#     """
#     The SHallow REcurrent Decoder (SHRED) Neural Network.
    
#     SHRED learns a mapping from trajectories of sensor measurements to a high-dimensional, spatio-temporal state.

#     Attributes:
#     -----------
#     sequence : str, optional
#         The sequence model used in SHRED (optional).
#         Choose from:
#             * 'LSTM': Long-short term memory model (default)
#     decoder : str, optional
#         The decoder model used in SHRED (optional).
#         Choose from:
#             * 'SDN': Shallow decoder model (default)
    
#     Methods:
#     --------
#     fit(data, sensors = 3, compressed = True, batch_size=64, num_epochs=4000, lr=1e-3, verbose=True, patience=5):
#         Train SHRED using the high-dimensional state space data.
    
#     recon(sensors):
#         Reconstruct the high-dimensional state space from the provided sensor measurements.

#     forecast(n):
#         Forecast the high-dimensional state space for `n` timesteps into the future.

#     """

#     def __init__(self, sequence, decoder):
#         """
#         Initialize SHRED with sequence model and decoder model.
#         """
#         super().__init__()
#         self._sequence_str = sequence.model_name
#         self._sequence_model = sequence
#         self._decoder_str = decoder.model_name
#         self._decoder_model = decoder
#         self._best_L2_error = None

#     def forward(self, x):
#         h_out = self._sequence_model(x)
#         output = self._decoder_model(h_out)
#         return output
    
#     def fit(self,model, train_dataset, val_dataset, num_sensors, output_size, batch_size, num_epochs, lr, verbose, patience):
#         """
#         Train SHRED using the high-dimensional state space data.

#         Parameters:
#         -----------
#         batch_size : int, optional
#             Number of samples per batch for training. Default is 64.

#         num_epochs : int, optional
#             Number of epochs for training the model. Default is 4000.

#         lr : float, optional
#             Learning rate for the optimizer. Default is 1e-3.

#         verbose : bool, optional
#             If True, prints progress during training. Default is True.

#         patience : int, optional
#             Number of epochs to wait for improvement before early stopping. Default is 5.
        
#         """        
#         ########################################### CONFIGURE SHRED MODEL ###############################################
#         # self._sequence_model = self.SEQUENCE_MODELS[self._sequence_str](input_size = num_sensors)
#         # sequence_out_size = self._sequence_model.hidden_size # hidden/latent size (output size of sequence model)
#         # self._decoder_model = self.DECODER_MODELS[self._decoder_str](input_size=sequence_out_size, output_size=output_size)
#         ############################################ SHRED TRAINING #####################################################
#         train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
#         criterion = torch.nn.MSELoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#         val_error_list = []
#         patience_counter = 0
#         best_params = model.state_dict()
#         for epoch in range(1, num_epochs + 1):
#             model.train()
#             running_loss = 0.0
#             running_error = 0.0
#             if verbose:
#                 pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{num_epochs}', unit='batch')
#             for inputs, target in train_loader:
#                 outputs = model(inputs)
#                 optimizer.zero_grad()
#                 loss = criterion(outputs, target)
#                 loss.backward()
#                 optimizer.step()
#                 running_loss += loss.item()
#                 train_error = torch.linalg.norm(outputs - target) / torch.linalg.norm(target)
#                 running_error += train_error.item()

#                 if verbose:
#                     pbar.set_postfix({
#                         'loss': running_loss / (pbar.n + 1),  # Average train loss
#                         'L2': running_error / (pbar.n + 1)  # Average train error
#                     })
#                     pbar.update(1)

#             model.eval()
#             with torch.no_grad():
#                 val_outputs = model(val_dataset.X)
#                 val_loss = criterion(val_outputs, val_dataset.Y).item()
#                 val_error = torch.linalg.norm(val_outputs - val_dataset.Y) / torch.linalg.norm(val_dataset.Y)
#                 val_error = val_error.item()
#                 val_error_list.append(val_error)

#             if verbose:
#                 pbar.set_postfix({
#                     'loss': running_loss / len(train_loader),
#                     'L2': running_error / len(train_loader),
#                     'val_loss': val_loss,
#                     'val_L2': val_error
#                 })
#                 pbar.close()

#             if patience is not None:
#                 if val_error == torch.min(torch.tensor(val_error_list)):
#                     patience_counter = 0
#                     self._best_L2_error = val_error
#                     best_params = model.state_dict()
#                 else:
#                     patience_counter += 1
#                 if patience_counter == patience:
#                     print("Early stopping triggered: patience threshold reached.")
#                     model.load_state_dict(best_params)
#                     return torch.tensor(val_error_list).cpu()
        
#         if patience is None:
#             self._best_L2_error = val_error

#         model.load_state_dict(best_params)
#         return torch.tensor(val_error_list).detach().cpu().numpy()
