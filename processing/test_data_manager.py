"""
SHREDDataManager Module
========================

This module contains the `SHREDDataManager` class, which orchestrates 
`SHREDDataProcessor` objects for managing, preprocessing, and postprocessing
datasets for reconstructor and forecastor tasks.

Classes
-------
SHREDDataManager
"""

import torch
from .utils import *
from .data_processor import *

class SHREDDataManager:
    """
    Manages SHREDDataProcessor objects to prepare datasets for machine learning.

    The `SHREDDataManager` class orchestrates the creation, preprocessing, 
    and postprocessing of datasets required for SHRED reconstructor and 
    forecastor models. It supports flexible configurations for scaling, 
    dimensionality reduction, and dataset splitting.

    Attributes
    ----------
    scaling : str
        Scaling method to use ('minmax', 'standard').
    compression : bool
        Whether to apply dimensionality reduction.
    time : array-like, optional
        1D numpy array of timestamps.
    lags : int
        Number of time steps to look back.
    train_size : float
        Proportion of data to use for training.
    val_size : float
        Proportion of data to use for validation.
    test_size : float
        Proportion of data to use for testing.
    reconstructor_flag : bool
        Flag to indicate whether to generate datasets for the reconstructor.
    forecastor_flag : bool
        Flag to indicate whether to generate datasets for the forecastor.

    Methods
    -------
    add(data, random_sensors=None, stationary_sensors=None, mobile_sensors=None, compression=True, id=None, scaling="minmax", time=None)
        Creates and adds a new `SHREDDataProcessor` object.
    preprocess()
        Generates train, validation, and test `SHREDDataset` objects.
    """

    def __init__(self, lags=20, time=None, train_size=0.75, val_size=0.15, test_size=0.15, scaling="minmax", compression=True, reconstructor=True, forecastor=True):
        """
        Initializes the SHREDDataManager with configuration settings.

        Parameters
        ----------
        lags : int, optional
            Number of time steps to look back (default is 20).
        time : array-like, optional
            1D numpy array of timestamps.
        train_size : float, optional
            Proportion of data to use for training (default is 0.75).
        val_size : float, optional
            Proportion of data to use for validation (default is 0.15).
        test_size : float, optional
            Proportion of data to use for testing (default is 0.15).
        scaling : str, optional
            Scaling method ('minmax' or 'standard') (default is 'minmax').
        compression : bool, optional
            Whether to apply dimensionality reduction (default is True).
        reconstructor : bool, optional
            Whether to generate datasets for the reconstructor (default is True).
        forecastor : bool, optional
            Whether to generate datasets for the forecastor (default is True).
        """
        self.scaling = scaling
        self.compression = compression
        self.time = time
        self.lags = lags
        self.data_processors = []
        self.reconstructor_indices = None
        self.forecastor_indices = None
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

    def add(self, data, random_sensors=None, stationary_sensors=None, mobile_sensors=None, compression=True, id=None, scaling="minmax", time=None):
        """
        Creates and adds a new SHREDDataProcessor object.

        Parameters
        ----------
        data : array-like
            The input dataset.
        random_sensors : int, optional
            Number of randomly placed stationary sensors.
        stationary_sensors : list of tuple, optional
            Coordinates of stationary sensors.
        mobile_sensors : list of tuple, optional
            Coordinates for mobile sensors over time.
        compression : bool, optional
            Whether to apply dimensionality reduction (default is True).
        id : str, optional
            Unique identifier for the dataset.
        scaling : str, optional
            Scaling method ('minmax' or 'standard') (default is 'minmax').
        time : array-like, optional
            1D numpy array of timestamps.
        """
        ...

    def preprocess(self):
        """
        Generates train, validation, and test SHREDDataset objects.

        This method processes datasets for both reconstructor and forecastor tasks.
        It concatenates data from all SHREDDataProcessor objects and prepares
        `SHREDDataset` objects for training, validation, and testing.

        Returns
        -------
        tuple
            Train, validation, and test `SHREDDataset` objects.
        """
        ...
