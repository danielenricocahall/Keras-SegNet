import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
import numpy as np

from custom_layers.layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


def test_max_pooling_argmax():
    # GIVEN we have some dummy test data which is 1-4 arranged as a 2 x 2
    data = np.asarray([[
        [1, 2],
        [3, 4]]], np.float32)
    data = np.expand_dims(data, axis=-1)
    tensor = tf.convert_to_tensor(data, np.float32)

    # WHEN we supply this data to our custom layer
    inp = Input(shape=(2, 2, 1))
    out = MaxPoolingWithArgmax2D()(inp)
    model = Model(inp, out)
    result = model.predict([tensor])

    # THEN the output should contain the index of the maximum argument (3), and the maximum argument (4)
    assert result[0][0].tolist() == [[[4.0]]]
    assert result[1][0].tolist() == [[[3.0]]]


def test_max_unpooling():
    # GIVEN we have some dummy test data which is 1-4 arranged as a 2 x 2
    data = np.asarray([[
        [1, 2],
        [3, 4]]], np.float32)
    data = np.expand_dims(data, axis=-1)
    tensor = tf.convert_to_tensor(data, np.float32)

    # WHEN we supply this data to our custom layers
    inp = Input(shape=(2, 2, 1))
    pool_1, mask_1 = MaxPoolingWithArgmax2D()(inp)
    out = MaxUnpooling2D()([pool_1, mask_1])
    model = Model(inp, out)
    result = model.predict([tensor])

    # THEN the output should be a sparse version of our input (only the maximum argument is retained)
    assert result.tolist()[0] == [[[0.0], [0.0]], [[0.0], [4.0]]]

