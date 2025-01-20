import numpy as np

def mean_squared_error(predictions, targets):
    """
    Calculate the Mean Squared Error (MSE).

    Parameters:
    -----------
    predictions : np.ndarray
        Predicted values.
    targets : np.ndarray
        Ground truth values.

    Returns:
    --------
    float
        The MSE value.
    """
    return np.mean((predictions - targets) ** 2)

def mean_absolute_error(predictions, targets):
    """
    Calculate the Mean Absolute Error (MAE).

    Parameters:
    -----------
    predictions : np.ndarray
        Predicted values.
    targets : np.ndarray
        Ground truth values.

    Returns:
    --------
    float
        The MAE value.
    """
    return np.mean(np.abs(predictions - targets))

def custom_metric(predictions, targets):
    """
    Placeholder for a custom metric.

    Parameters:
    -----------
    predictions : np.ndarray
        Predicted values.
    targets : np.ndarray
        Ground truth values.

    Returns:
    --------
    float
        Custom metric value.
    """
    # Implement your own custom metric logic
    return np.sum(predictions - targets) / len(predictions)
