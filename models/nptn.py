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


def two_layer_nptn(input_shape, num_classes, out_channels, num_transforms):
    """Two-layer NPTN as described in the paper.

    out_channels and num_transforms refer to the first layer only.
    """
    model = Sequential()

    layer_1 = lambda x: nptn_layer(x, 3, 3, out_channels, num_transforms)
    layer_2 = lambda x: nptn_layer(x, 3, out_channels, 16, 1)

    model.add(Lambda(layer_1, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Lambda(layer_2))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    return model
