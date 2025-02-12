from .abstract_sequence import AbstractSequence
import torch.nn as nn
import torch

class Transformer(AbstractSequence):
    """
    Transformer-based sequence model.
    """

    def __init__(self, input_size=None, d_model=64, num_encoder_layers=2, nhead=8, dropout=0.1):
        """
        Lazily initialize the Transformer model.

        Parameters:
        -----------
        input_size : int, optional
            The size of the input features. Default is None.
        d_model : int, optional
            The size of the hidden state (embedding dimensions). Default is 64.
        num_encoder_layers : int, optional
            The number of Transformer encoder layers. Default is 2.
        nhead : int, optional
            The number of attention heads. Default is 8.
        dropout : float, optional
            Dropout probability. Default is 0.1.
        """
        super().__init__(input_size=input_size, output_size=d_model)
        self.input_size = input_size
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.nhead = nhead
        self.dropout = dropout
        self.encoder = None
        self.input_projection = None
        self.is_initialized = False

    def initialize(self, input_size):
        """
        Initialize the Transformer model with the input size.

        Parameters:
        -----------
        input_size : int
            The size of the input features.
        """
        super().initialize(input_size)

        # Add a projection layer if input_size does not match d_model
        if input_size != self.d_model:
            self.input_projection = nn.Linear(input_size, self.d_model)
        
        # Create Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)

    def forward(self, x, mask=None):
        """
        Forward pass through the Transformer encoder.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size).
        mask : torch.Tensor, optional
            Attention mask for padding or future masking. Shape: 
            (batch_size, sequence_length). Default is None.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, d_model), representing
            the final hidden state of the last time step.
        """
        super().forward(x)

        # Project input if needed
        if self.input_projection is not None:
            x = self.input_projection(x)

        # Ensure mask is on the same device
        if mask is not None and mask.device != x.device:
            mask = mask.to(x.device)

        # Pass through Transformer Encoder
        encoder_output = self.encoder(x, src_key_padding_mask=mask)

        # Return the last hidden state
        return encoder_output[:, -1, :]
    
    @property
    def model_name(self):
        return "Transformer"
