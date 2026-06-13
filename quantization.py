import numpy as np


def quantize_int8(parameters):
    """
    Convert float32 model parameters to int8.
    Returns quantized parameters and scale values.
    """

    quantized_params = []
    scales = []

    for param in parameters:
        max_val = np.max(np.abs(param))

        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / 127

        q_param = np.round(param / scale).astype(np.int8)

        quantized_params.append(q_param)
        scales.append(scale)

    return quantized_params, scales


def dequantize_int8(quantized_params, scales):
    """
    Convert int8 parameters back to float32.
    """

    dequantized_params = []

    for q_param, scale in zip(quantized_params, scales):
        param = q_param.astype(np.float32) * scale
        dequantized_params.append(param)

    return dequantized_params


def calculate_size_bytes(parameters):
    """
    Calculate total size of parameters in bytes.
    """

    return sum(param.nbytes for param in parameters)