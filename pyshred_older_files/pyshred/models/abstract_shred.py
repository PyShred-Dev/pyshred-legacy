from abc import ABC, abstractmethod
import torch.nn as nn

class AbstractSHRED(ABC, nn.Module):
    """
    Abstract base class for SHRED-like models.
    Defines the interface all SHRED implementations must follow.
    """
    
    def __init__(self):
        super().__init__()
        self._sequence_module = None
        self._decoder_module = None
        self._is_fitted = False

    @abstractmethod
    def fit(self, data, sensors, *args, **kwargs):
        """
        Train the SHRED model using the provided data and sensors.
        """
        pass

    @abstractmethod
    def predict(self, start, end, sensor_data=None, sensor_data_time=None):
        """
        Predict the state space or sensor values over a given time range.
        """
        pass

    @abstractmethod
    def recon(self, sensor_measurements):
        """
        Perform state-space reconstruction using sensor measurements.
        """
        pass

    @abstractmethod
    def forecast(self, timesteps, sensor_data=None, sensor_data_time=None):
        """
        Forecast the state space for a given number of timesteps into the future.
        """
        pass

    @abstractmethod
    def summary(self):
        """
        Print a summary of the SHRED model.
        """
        pass

    @abstractmethod
    def _scale_sensor_data(self, unscaled_sensor_data):
        """
        Scale sensor data before feeding it into the model.
        """
        pass

    @abstractmethod
    def _unscale_sensor_data(self, scaled_sensor_data):
        """
        Unscale sensor data after model processing.
        """
        pass
