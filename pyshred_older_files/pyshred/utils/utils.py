import numpy as np

def generate_lagged_sequences(data, lags):
    """
    Generate lagged sequences from time series data.

    Parameters:
    - data: array-like, shape (n_timesteps, n_features)
        The raw time series data.
    - lags: int
        Number of lagged timesteps to include.

    Returns:
    - lagged_data: array-like, shape (n_samples, lags, n_features)
        The generated lagged sequences.
    """
    if len(data) < lags:
        raise ValueError("Number of timesteps in data must be greater than or equal to the number of lags.")
    
    lagged_data = [data[i:i + lags] for i in range(len(data) - lags + 1)]
    return np.array(lagged_data)
