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
# Support:
# 4-bit
# 6-bit
# 8-bit
# =========================================

import numpy as np


def quantize_uniform_for_flower(
    parameters,
    bits,
):
    """
    Symmetric Uniform Quantization

    Parameters:

        parameters:
            float32 model parameters

        bits:
            quantization bit-width

    Returns:

        flower_quantized_params:
            float32 carrier for Flower

        q_params:
            quantized integer arrays

        scales:
            scale for each tensor
    """

    flower_quantized_params = []

    q_params = []

    scales = []

    qmax = (

        2 ** (bits - 1)

    ) - 1

    qmin = -(

        2 ** (bits - 1)

    )

    for param in parameters:

        max_val = np.max(

            np.abs(param)

        )

        # Avoid divide by zero

        if max_val == 0:

            scale = 1.0

        else:

            scale = (

                max_val

                /

                qmax

            )

        q = np.round(

            param

            /

            scale

        )

        q = np.clip(

            q,

            qmin,

            qmax,

        )

        q = q.astype(

            np.int32

        )

        q_params.append(

            q

        )

        scales.append(

            float(scale)

        )

        # Flower does not support int4/int6/int8

        # Use float32 carrier

        flower_quantized_params.append(

            q.astype(

                np.float32

            )

        )

    return (

        flower_quantized_params,

        q_params,

        scales,

    )


# =========================================
# Adaptive Dequantization
# =========================================

def dequantize_uniform_from_flower(

    flower_params,

    scales,

    bits,

):
    """
    Recover float32 parameters from
    Flower float32 carrier.

    Parameters:

        flower_params:

            float32 carrier

        scales:

            quantization scales

        bits:

            quantization bit-width

    Returns:

        restored_params

    """

    restored_params = []

    qmax = (

        2 ** (bits - 1)

    ) - 1

    qmin = -(

        2 ** (bits - 1)

    )

    for param, scale in zip(

        flower_params,

        scales,

    ):

        q = np.round(

            param

        )

        q = np.clip(

            q,

            qmin,

            qmax,

        )

        q = q.astype(

            np.float32

        )

        restored = (

            q

            *

            scale

        )

        restored_params.append(

            restored.astype(

                np.float32

            )

        )

    return restored_params


# =========================================
# Adaptive Communication Cost
# =========================================

def calculate_quantized_payload_size_bytes(

    q_params,

    scales,

):
    """
    Calculate logical payload size.

    Quantized weights:

        int32 arrays are used in Python,

        but communication cost is calculated

        according to logical bit-width.

    """

    total_bytes = 0

    for q, scale in zip(

        q_params,

        scales,

    ):

        bits = int(

            np.ceil(

                np.log2(

                    np.max(

                        np.abs(q)

                    )

                    +

                    1

                )

            )

        )

        bits = max(

            bits + 1,

            4,

        )

        payload_bits = (

            q.size

            *

            bits

        )

        payload_bytes = (

            payload_bits

            /

            8

        )

        total_bytes += payload_bytes

        # scale stored as float32

        total_bytes += 4

    return int(total_bytes)