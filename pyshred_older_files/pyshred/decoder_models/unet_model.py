import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract_decoder import AbstractDecoder

class UNet1D(AbstractDecoder):
    """
    A very simplified 1D U-Net for decoding a latent (1D) representation into
    a higher-dimensional flattened space. In practice, adapt for 2D or 3D as needed.
    """

    def __init__(self, input_size=None, output_size=None, base_channels=64):
        super().__init__(input_size, output_size)
        self.base_channels = base_channels

        # The following layers will be created in initialize(...)
        self.enc_conv1 = None
        self.enc_conv2 = None
        self.dec_conv1 = None
        self.dec_conv2 = None
        self.final_conv = None

    def initialize(self, input_size, output_size):
        """
        Called once we know the input_size (latent dimension) and output_size
        (final flattened dimension).
        """
        super().initialize(input_size, output_size)

        # For 1D, treat the latent dimension as 'channels' of length input_size
        # Here we pretend we have a "signal" of length 1 with 'input_size' channels.
        in_channels = input_size
        out_channels = self.base_channels
        
        # Down
        self.enc_conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.enc_conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels*2, out_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Up
        self.dec_conv1 = nn.Sequential(
            nn.Conv1d(out_channels*2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # final output
        self.final_conv = nn.Conv1d(out_channels, output_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. 
        x shape: (batch_size, input_size)  [since the SHRED sequence models produce (batch, latent_dim)]
        We want to interpret x as (batch, channels, length=1).
        """
        batch_size = x.size(0)
        # reshape to (batch, in_channels, length=1)
        x = x.view(batch_size, self.input_size, 1)

        # ENCODER
        x1 = self.enc_conv1(x)  # shape: (batch, base_channels, 1)
        # just a simple downsample by factor of 2
        x1_pool = F.avg_pool1d(x1, kernel_size=2, stride=2, ceil_mode=True)  # shape: (batch, base_channels, ceil(1/2))

        x2 = self.enc_conv2(x1_pool)  # shape: (batch, base_channels*2, ???)
        x2_pool = F.avg_pool1d(x2, kernel_size=2, stride=2, ceil_mode=True)  # further reduce dimension

        # DECODER (just a simple example upsample)
        x2_up = F.interpolate(x2_pool, scale_factor=2, mode='nearest')
        x3 = self.dec_conv1(x2_up)  # shape: (batch, base_channels, ???)

        x3_up = F.interpolate(x3, scale_factor=2, mode='nearest')

        # final output channels
        out = self.final_conv(x3_up)  # shape: (batch, output_size, ???)

        # we want the final shape (batch, output_size) if the “length” is 1
        # but with multiple pooling layers and small input=1, you might end up with length > 1 or 0.
        # This is just a minimal example – adapt kernel sizes carefully for real data.
        
        # For safety, flatten: (batch, output_size*length)
        # If you have undone flattening in SHRED (like PDE/2D data), adapt accordingly.
        out = out.view(batch_size, -1)  # shape: (batch, output_size * length)
        return out

    @property
    def model_name(self):
        return "UNET"
