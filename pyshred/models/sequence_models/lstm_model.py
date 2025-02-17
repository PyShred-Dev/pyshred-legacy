import torch.nn as nn
import torch
from .abstract_sequence import AbstractSequence


class LSTM(AbstractSequence):


    def __init__(self, hidden_size:int =64, num_layers:int =2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = None # lazy initialization
        self.output_size = hidden_size


    def initialize(self, input_size:int, **kwargs):
        super().initialize(input_size)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )


    def forward(self, x):
        """
        Forward pass through the LSTM model.
        """
        super().forward(x)
        device = next(self.parameters()).device
        # Initialize hidden and cell
        h_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=device)
        c_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=device)
        out, (h_out, c_out) = self.lstm(x, (h_0, c_0))

        return {
            "sequence_output": out,
            "final_hidden_state": h_out[-1].view(-1, self.hidden_size)
        }
    
    @property
    def model_name(self):
        return "LSTM"