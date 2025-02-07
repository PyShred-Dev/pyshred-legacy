# class _SHRED_RECONSTRUCTOR(nn.Module):
# import sys
# import pandas as pd
import torch.nn as nn
# from sklearn.utils.extmath import randomized_svd
# import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
# import pickle
import torch
from processing.utils import l2

class RECONSTRUCTOR(nn.Module):
    """
    The SHallow REcurrent Decoder (SHRED) Neural Network.
    
    SHRED learns a mapping from trajectories of sensor measurements to a high-dimensional, spatio-temporal state.

    Attributes:
    -----------
    sequence : str, optional
        The sequence model used in SHRED (optional).
        Choose from:
            * 'LSTM': Long-short term memory model (default)
    decoder : str, optional
        The decoder model used in SHRED (optional).
        Choose from:
            * 'SDN': Shallow decoder model (default)
    
    Methods:
    --------
    fit(data, sensors = 3, compressed = True, batch_size=64, num_epochs=4000, lr=1e-3, verbose=True, patience=5):
        Train SHRED using the high-dimensional state space data.
    
    recon(sensors):
        Reconstruct the high-dimensional state space from the provided sensor measurements.

    forecast(n):
        Forecast the high-dimensional state space for `n` timesteps into the future.

    """

    def __init__(self, sequence, decoder):
        """
        Initialize SHRED with sequence model and decoder model.
        """
        super().__init__()
        self._sequence_str = sequence.model_name
        self._sequence_model = sequence
        self._decoder_str = decoder.model_name
        self._decoder_model = decoder
        self._best_L2_error = None

    def forward(self, x):
        h_out = self._sequence_model(x)
        output = self._decoder_model(h_out)
        return output
    
    def fit(self,model, train_dataset, val_dataset, num_sensors, output_size, batch_size, num_epochs, lr, verbose, patience):
        """
        Train SHRED using the high-dimensional state space data.

        Parameters:
        -----------
        batch_size : int, optional
            Number of samples per batch for training. Default is 64.

        num_epochs : int, optional
            Number of epochs for training the model. Default is 4000.

        lr : float, optional
            Learning rate for the optimizer. Default is 1e-3.

        verbose : bool, optional
            If True, prints progress during training. Default is True.

        patience : int, optional
            Number of epochs to wait for improvement before early stopping. Default is 5.
        
        """        
        ########################################### CONFIGURE SHRED MODEL ###############################################
        # self._sequence_model = self.SEQUENCE_MODELS[self._sequence_str](input_size = num_sensors)
        # sequence_out_size = self._sequence_model.hidden_size # hidden/latent size (output size of sequence model)
        # self._decoder_model = self.DECODER_MODELS[self._decoder_str](input_size=sequence_out_size, output_size=output_size)
        ############################################ SHRED TRAINING #####################################################
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        val_error_list = []
        patience_counter = 0
        best_params = model.state_dict()
        best_val_error = float('inf')  # Initialize with a large value
        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0
            running_error = 0.0
            if verbose:
                pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{num_epochs}', unit='batch')
            for inputs, target in train_loader:
                outputs = model(inputs)
                optimizer.zero_grad()
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_error = l2(target, outputs)
                running_error += train_error.item()

                if verbose:
                    pbar.set_postfix({
                        'loss': running_loss / (pbar.n + 1),  # Average train loss
                        'L2': running_error / (pbar.n + 1)  # Average train error
                    })
                    pbar.update(1)

            model.eval()
            with torch.no_grad():
                val_outputs = model(val_dataset.X)
                val_loss = criterion(val_outputs, val_dataset.Y).item()
                val_error = l2(val_dataset.Y, val_outputs)
                val_error = val_error.item()
                val_error_list.append(val_error)

            if verbose:
                pbar.set_postfix({
                    'loss': running_loss / len(train_loader),
                    'L2': running_error / len(train_loader),
                    'val_loss': val_loss,
                    'val_L2': val_error
                })
                pbar.close()

            # Update best model weights if the val error improves
            if val_error < best_val_error:
                best_val_error = val_error
                best_params = model.state_dict()  # Save best model parameters
                self._best_L2_error = val_error
                patience_counter = 0  # Reset patience counter if improvement occurs
            else:
                patience_counter += 1

            # Early stopping logic
            if patience is not None and patience_counter == patience:
                print("Early stopping triggered: patience threshold reached.")
                break  # Exit training loop

        model.load_state_dict(best_params)
        return torch.tensor(val_error_list).detach().cpu().numpy()
