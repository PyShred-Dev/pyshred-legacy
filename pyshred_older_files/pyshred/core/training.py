import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(
    model,
    train_dataset,
    val_dataset=None,
    batch_size=64,
    num_epochs=100,
    lr=1e-3,
    criterion=None,
    optimizer=None,
    patience=10,
    verbose=True,
):
    """
    Train a given model using the provided datasets.

    Parameters:
    -----------
    model : nn.Module
        The PyTorch model to train.
    train_dataset : Dataset
        The training dataset.
    val_dataset : Dataset, optional
        The validation dataset, used for early stopping.
    batch_size : int, optional
        Number of samples per batch. Default is 64.
    num_epochs : int, optional
        Maximum number of epochs. Default is 100.
    lr : float, optional
        Learning rate for the optimizer. Default is 1e-3.
    criterion : callable, optional
        Loss function. Default is `torch.nn.MSELoss`.
    optimizer : torch.optim.Optimizer, optional
        Optimizer for the training process. Default is Adam.
    patience : int, optional
        Number of epochs to wait for improvement before early stopping. Default is 10.
    verbose : bool, optional
        Whether to print progress. Default is True.

    Returns:
    --------
    dict
        A dictionary containing training loss, validation loss, and model state.
    """
    # Default loss function
    if criterion is None:
        criterion = torch.nn.MSELoss()

    # Default optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
    )

    # Early stopping variables
    best_val_loss = float("inf")
    best_model_state = model.state_dict()
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0

        if verbose:
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}")

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if verbose:
                pbar.set_postfix({"train_loss": running_loss / (pbar.n + 1)})
                pbar.update(1)

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        if verbose:
            pbar.close()

        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(model.device), targets.to(model.device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if verbose:
                print(f"Validation Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print("Early stopping triggered.")
                break

    # Restore the best model state
    model.load_state_dict(best_model_state)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses if val_loader else None,
        "best_model_state": best_model_state,
    }
