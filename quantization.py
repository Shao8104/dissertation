import pickle
import numpy as np


def quantize_int8(parameters):
    """
    Quantize float32 parameters to int8.

    Returns:
        quantized_params: List[np.ndarray], int8 tensors
        scales: List[float], one scale per tensor
    """

    quantized_params = []
    scales = []

    for param in parameters:
        param = param.astype(np.float32)

        max_val = np.max(np.abs(param))

        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / 127.0

        q_param = np.round(param / scale)
        q_param = np.clip(q_param, -128, 127)
        q_param = q_param.astype(np.int8)

        quantized_params.append(q_param)
        scales.append(float(scale))

    return quantized_params, scales


def dequantize_int8(quantized_params, scales):
    """
    Dequantize int8 parameters back to float32.
    """

    dequantized_params = []

    for q_param, scale in zip(quantized_params, scales):
        param = q_param.astype(np.float32) * scale
        dequantized_params.append(param.astype(np.float32))

    return dequantized_params


def quantize_int8_for_flower(parameters):
    """
    Quantize float32 parameters for Flower transport.

    Flower NumPyClient expects List[np.ndarray].
    To keep the same tensor structure, int8 values are carried as float32 arrays.

    Actual communication cost should be calculated from the original int8 tensors,
    not from the float32 carrier tensors.
    """

    quantized_params, scales = quantize_int8(parameters)

    flower_params = [
        q_param.astype(np.float32)
        for q_param in quantized_params
    ]

    return flower_params, quantized_params, scales


def dequantize_int8_from_flower(flower_params, scales):
    """
    Convert Flower-carried quantized values back to float32 model parameters.

    flower_params are float32 arrays containing integer values in [-128, 127].
    """

    quantized_params = [
        np.round(param).astype(np.int8)
        for param in flower_params
    ]

    return dequantize_int8(quantized_params, scales)


def serialize_scales(scales):
    """
    Serialize scale values so they can be passed through Flower metrics.
    """

    return pickle.dumps(scales)


def deserialize_scales(serialized_scales):
    """
    Deserialize scale values from Flower metrics.
    """

    return pickle.loads(serialized_scales)


def calculate_size_bytes(parameters):
    """
    Calculate total size of a list of numpy arrays in bytes.
    """

    return sum(param.nbytes for param in parameters)


def calculate_int8_payload_size_bytes(quantized_params, scales):
    """
    Calculate the logical communication size of int8 parameters plus scales.
    """

    q_size = calculate_size_bytes(quantized_params)

    scale_arrays = [
        np.array([scale], dtype=np.float32)
        for scale in scales
    ]

    scale_size = calculate_size_bytes(scale_arrays)

    return q_size + scale_size


def calculate_reduction(original_params, quantized_params, scales):
    """
    Calculate communication reduction from float32 to int8 + scales.
    """

    original_size = calculate_size_bytes(original_params)

    quantized_size = calculate_int8_payload_size_bytes(
        quantized_params,
        scales,
    )

    reduction = (
        1.0 - quantized_size / original_size
    ) * 100.0

    return original_size, quantized_size, reduction

# =========================================
# Adaptive Uniform Quantization
# Support 4-bit / 6-bit / 8-bit
# =========================================

def quantize_uniform_for_flower(parameters, bits):
    """
    Quantize float32 parameters using symmetric uniform quantization.

    Fixed INT8:
        bits = 8

    Adaptive:
        bits = 4, 6, or 8
    """

    if bits not in [4, 6, 8]:
        raise ValueError(
            f"Unsupported bit-width: {bits}. "
            "Only 4, 6, and 8 bits are supported."
        )

    quantized_params = []
    flower_params = []
    scales = []

    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))

    for param in parameters:
        param = param.astype(np.float32)

        max_val = np.max(np.abs(param))

        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / qmax

        q_param = np.round(param / scale)
        q_param = np.clip(q_param, qmin, qmax)
        q_param = q_param.astype(np.int32)

        quantized_params.append(q_param)
        scales.append(float(scale))

        # Flower NumPyClient transports ndarrays.
        # We use float32 as carrier, but logical cost is calculated by bit-width.
        flower_params.append(q_param.astype(np.float32))

    return flower_params, quantized_params, scales


def dequantize_uniform_from_flower(flower_params, scales, bits):
    """
    Dequantize Flower-carried quantized parameters back to float32.
    """

    if bits not in [4, 6, 8]:
        raise ValueError(
            f"Unsupported bit-width: {bits}. "
            "Only 4, 6, and 8 bits are supported."
        )

    dequantized_params = []

    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))

    for q_param, scale in zip(flower_params, scales):
        q_param = np.round(q_param)
        q_param = np.clip(q_param, qmin, qmax)
        q_param = q_param.astype(np.float32)

        param = q_param * scale
        dequantized_params.append(param.astype(np.float32))

    return dequantized_params


def calculate_uniform_payload_size_bytes(quantized_params, scales, bits):
    """
    Calculate logical communication payload size for adaptive quantization.

    Note:
        The arrays are stored as int32 in Python for convenience,
        but the communication cost is calculated using the selected bit-width.
    """

    if bits not in [4, 6, 8]:
        raise ValueError(
            f"Unsupported bit-width: {bits}. "
            "Only 4, 6, and 8 bits are supported."
        )

    total_bits = 0

    for q_param in quantized_params:
        total_bits += q_param.size * bits

    weight_bytes = total_bits / 8

    # one float32 scale per tensor
    scale_bytes = len(scales) * 4

    return int(weight_bytes + scale_bytes)


def calculate_uniform_reduction(original_params, quantized_params, scales, bits):
    """
    Calculate upload communication reduction compared with float32 upload.
    """

    original_size = calculate_size_bytes(original_params)

    quantized_size = calculate_uniform_payload_size_bytes(
        quantized_params,
        scales,
        bits,
    )

    reduction = (
        1.0 - quantized_size / original_size
    ) * 100.0

    return original_size, quantized_size, reduction
# =========================================
# Serialize / Deserialize Scales
# =========================================

def serialize_scales(scales):
    """
    Convert scales list to string for Flower metrics.
    """

    return ",".join(

        [

            str(s)

            for s in scales

        ]

    )


def deserialize_scales(serialized_scales):
    """
    Recover scales list from string.
    """

    return [

        float(s)

        for s in serialized_scales.split(",")

    ]