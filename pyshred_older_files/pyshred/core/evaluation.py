import torch
import numpy as np


def evaluate_model(model, dataset, batch_size=64, criterion=None):
    """
    Evaluate a model on a given dataset.

    Parameters:
    -----------
    model : nn.Module
        The PyTorch model to evaluate.
    dataset : Dataset
        The dataset to evaluate on.
    batch_size : int, optional
        Number of samples per batch. Default is 64.
    criterion : callable, optional
        Loss function for evaluation. Default is `torch.nn.MSELoss`.

    Returns:
    --------
    dict
        A dictionary containing the average loss and predictions.
    """
    if criterion is None:
        criterion = torch.nn.MSELoss()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model.eval()

    total_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions.append(outputs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    return {"loss": avg_loss, "predictions": predictions, "targets": targets}


def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics for predictions and targets.

    Parameters:
    -----------
    predictions : np.ndarray
        Predicted values.
    targets : np.ndarray
        Ground truth values.

    Returns:
    --------
    dict
        A dictionary containing evaluation metrics.
    """
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))

    return {"MSE": mse, "MAE": mae}
