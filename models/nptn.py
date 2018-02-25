from keras.layers import (BatchNormalization, Dense, Flatten, Lambda,
                          MaxPooling2D)
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
import tensorflow as tf


def nptn_layer(input, kernel_size, in_channels, out_channels, num_transforms):
    """NPTN layer
    use separate scopes to use multiple NPTN layers

    Biased.

    input
    kernel_size is k
    in_channels is M
    out_channels is N
    num_transforms is |G|

    TODO
    """
    assert len(input.shape) == 4

    filter_shape = (kernel_size, kernel_size, in_channels,
                    out_channels * num_transforms)
    filter = tf.get_variable('weights', filter_shape, dtype=input.dtype)
    bias = tf.get_variable('bias', (1,), dtype=input.dtype)

    # Step 1: Convolution
    # The output of the convolution should have all outputs for a particular
    # input channel grouped together
    depthwise_out = tf.nn.depthwise_conv2d(input, filter, [1, 1, 1, 1], 'SAME')

    # Step 2: "Volumetric max pooling" across transformations
    splits = tf.split(depthwise_out, in_channels * out_channels, axis=3)
    max_splits = [tf.reduce_max(s, axis=3, keep_dims=True) for s in splits]

    # Steps 3 and 4: Reordering and "volumetric mean pooling"
    outputs = []
    for i in range(out_channels):
        gathered = tf.concat(max_splits[i::out_channels], 3)
        outputs.append(tf.reduce_mean(gathered, axis=3, keep_dims=True))
    output = tf.concat(outputs, 3)

    return output + bias


def two_layer_nptn(x, num_classes, out_channels, num_transforms):
    """Two-layer NPTN as described in the paper.

    out_channels and num_transforms refer to the first layer only.
    """
    with tf.variable_scope('nptn_1'):
        x = nptn_layer(x, 3, 3, out_channels, num_transforms)
        x = tf.layers.batch_normalization(x)
        x = parametric_relu(x)
        x = tf.layers.max_pooling2d(x, (2, 2), 2)

    with tf.variable_scope('nptn_2'):
        x = nptn_layer(x, 3, out_channels, 16, 1)
        x = tf.layers.batch_normalization(x)
        x = parametric_relu(x)
        x = tf.layers.max_pooling2d(x, (2, 2), 2)

    x = tf.contrib.layers.flatten(x)
    logits = tf.layers.dense(x, units=num_classes)

    return logits


def parametric_relu(_x):
    """Source: https://stackoverflow.com/a/40264459/."""
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg
