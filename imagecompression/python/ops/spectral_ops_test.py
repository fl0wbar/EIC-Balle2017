"""Tests of spectral_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow as tf
import imagecompression.python.ops.spectral_ops as spectral_ops


class SpectralOpsTest(tf.test.TestCase):
    def test_irdft1_matrix(self):
        for shape in [(4,), (3,)]:
            size = shape[0]
            matrix = spectral_ops.irdft_matrix(shape)
            # Test that the matrix is orthonormal.
            result = tf.matmul(matrix, tf.transpose(matrix))
            with self.test_session() as sess:
                (result,) = sess.run([result])
                self.assertAllClose(result, np.identity(size))

    def test_irdft2_matrix(self):
        for shape in [(7, 4), (8, 9)]:
            size = shape[0] * shape[1]
            matrix = spectral_ops.irdft_matrix(shape)
            # Test that the matrix is orthonormal.
            result = tf.matmul(matrix, tf.transpose(matrix))
            with self.test_session() as sess:
                (result,) = sess.run([result])
                self.assertAllClose(result, np.identity(size))

    def test_irdft3_matrix(self):
        for shape in [(3, 4, 2), (6, 3, 1)]:
            size = shape[0] * shape[1] * shape[2]
            matrix = spectral_ops.irdft_matrix(shape)
            # Test that the matrix is orthonormal.
            result = tf.matmul(matrix, tf.transpose(matrix))
            with self.test_session() as sess:
                (result,) = sess.run([result])
                self.assertAllClose(result, np.identity(size))


if __name__ == "__main__":
    tf.test.main()
