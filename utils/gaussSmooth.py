# Sourced from https://github.com/fwillett/speechBCI/blob/main/NeuralDecoder/neuralDecoder/neuralSequenceDecoder.py

import tensorflow as tf
from scipy.ndimage import gaussian_filter1d
import numpy as np

#@tf.function(experimental_relax_shapes=True)
def gaussSmooth(inputs, kernelSD=2, padding='SAME'):
    """
    Applies a 1D gaussian smoothing operation with tensorflow to smooth the data along the time axis.

    Args:
        inputs (tensor : B x T x N): A 3d tensor with batch size B, time steps T, and number of features N
        kernelSD (float): standard deviation of the Gaussian smoothing kernel

    Returns:
        smoothedData (tensor : B x T x N): A smoothed 3d tensor with batch size B, time steps T, and number of features N
    """

    #get gaussian smoothing kernel
    inp = np.zeros([100], dtype=np.float32)
    inp[50] = 1
    gaussKernel = gaussian_filter1d(inp, kernelSD)
    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel/np.sum(gaussKernel))

    # Apply depth_wise convolution
    B, T, C = inputs.shape.as_list()
    filters = tf.tile(gaussKernel[None, :, None, None], [1, 1, C, 1])  # [1, W, C, 1]
    inputs = inputs[:, None, :, :]  # [B, 1, T, C]
    smoothedInputs = tf.nn.depthwise_conv2d(inputs, filters, strides=[1, 1, 1, 1], padding=padding)
    smoothedInputs = tf.squeeze(smoothedInputs, 1)

    return smoothedInputs