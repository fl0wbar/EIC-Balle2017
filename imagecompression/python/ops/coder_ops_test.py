"""Coder operations tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf

from tensorflow.python.platform import test

import imagecompression.python.ops.coder_ops as coder_ops


class CoderOpsTest(test.TestCase):
    """Coder ops test.

    Coder ops have C++ tests. Python test just ensures that Python binding is not
    broken.
    """

    def testReadmeExample(self):
        data = tf.random_uniform((128, 128), 0, 10, dtype=tf.int32)
        histogram = tf.bincount(data, minlength=10, maxlength=10)
        cdf = tf.cumsum(histogram, exclusive=False)
        cdf = tf.pad(cdf, [[1, 0]])
        cdf = tf.reshape(cdf, [1, 1, -1])

        data = tf.cast(data, tf.int16)
        encoded = coder_ops.range_encode(data, cdf, precision=14)
        decoded = coder_ops.range_decode(encoded, tf.shape(data), cdf, precision=14)

        with self.test_session() as sess:
            self.assertAllEqual(*sess.run((data, decoded)))


if __name__ == "__main__":
    test.main()
