from sequence_models.lstm import LSTMSequence
from sequence_models.transformer import TransformerSequence
from decoder_models.sdn import SDNDecoder
from decoder_models.unet import UNetDecoder

# Model registry
SEQUENCE_MODELS = {
    "LSTM": LSTMSequence,
    "Transformer": TransformerSequence,
}

DECODER_MODELS = {
    "SDN": SDNDecoder,
    "UNet": UNetDecoder,
}

def get_sequence_model(name, **kwargs):
    """
    Retrieve a sequence model by name.

    Parameters:
    -----------
    name : str
        The name of the sequence model.
    kwargs : dict
        Additional arguments for model initialization.

    Returns:
    --------
    nn.Module
        Initialized sequence model.
    """
    if name not in SEQUENCE_MODELS:
        raise ValueError(f"Invalid sequence model '{name}'. Available options: {list(SEQUENCE_MODELS.keys())}.")
    return SEQUENCE_MODELS[name](**kwargs)

def get_decoder_model(name, **kwargs):
    """
    Retrieve a decoder model by name.

    Parameters:
    -----------
    name : str
        The name of the decoder model.
    kwargs : dict
        Additional arguments for model initialization.

    Returns:
    --------
    nn.Module
        Initialized decoder model.
    """
    if name not in DECODER_MODELS:
        raise ValueError(f"Invalid decoder model '{name}'. Available options: {list(DECODER_MODELS.keys())}.")
    return DECODER_MODELS[name](**kwargs)
