import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract_sequence import AbstractSequence

class TransformerSequence(AbstractSequence):
    """
    A minimal Transformer sequence model that can be plugged into SHRED in place of LSTM.
    """

    def __init__(
        self,
        d_model=64,   # dimensionality of the embedding
        nhead=8,      # number of heads in the multiheadattention models
        num_layers=2, # number of transformer encoder layers
        dim_feedforward=256, # feedforward network model dimension
        dropout=0.1,
        batch_first=True
    ):
        """
        Parameters
        ----------
        d_model : int
            Dimensionality of the embedding/hidden representation.
        nhead : int
            Number of parallel attention heads.
        num_layers : int
            Number of TransformerEncoder layers to stack.
        dim_feedforward : int
            Dimensionality of the feedforward layer in the TransformerEncoder.
        dropout : float
            Dropout probability.
        batch_first : bool
            If True, input and output tensors are provided as (batch, seq, feature).
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_first = batch_first
        self.output_size = d_model

        self.encoder_layer = None
        self.transformer_encoder = None
        self.input_embedding = None

    def initialize(self, input_size: int):
        """
        Called once we know the input_size (i.e. number of sensors).
        We create the embedding + Transformer layers.
        """
        super().initialize(input_size)
        # Embedding layer to project sensor measurements into d_model dimension
        self.input_embedding = nn.Linear(self.input_size, self.d_model)

        # Create a single (or multiple) TransformerEncoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=self.batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer.
        x is assumed to be of shape (batch_size, sequence_length, num_sensors)
        if batch_first is True.
        """
        super().forward(x)
        device = next(self.parameters()).device

        # Project input sensor data into an embedding
        embedded_input = self.input_embedding(x)  # shape: (batch_size, seq_len, d_model)

        # Pass through the Transformer encoder
        encoded_output = self.transformer_encoder(embedded_input)  
        # shape: (batch_size, seq_len, d_model)

        # We can pool (e.g., take the last timestep) or reduce along seq dimension
        # Here, we simply take the last time step for a single hidden representation
        # for each sample in the batch:
        # shape: (batch_size, d_model)
        # return encoded_output, encoded_output[:, -1, :]  # Returning full output and last token
        return {
            "sequence_output": encoded_output[:, -1, :],
            "final_hidden_state": encoded_output
        }

    @property
    def model_name(self):
        return "TRANSFORMER"
