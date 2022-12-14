"""Tests of padding ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow as tf
import imagecompression.python.ops.padding_ops as padding_ops


class PaddingOpsTest(tf.test.TestCase):
    def test_same_padding_corr(self):
        for ishape in [[10], [11]]:
            inputs = np.zeros(ishape, dtype=np.float32)
            inputs[len(inputs) // 2] = 1
            for kshape in [[4], [5]]:
                kernel = np.zeros(kshape, dtype=np.float32)
                kernel[len(kernel) // 2] = 1
                outputs = tf.nn.convolution(
                    tf.reshape(inputs, (1, 1, -1, 1)),
                    tf.reshape(kernel, (1, -1, 1, 1)),
                    padding="VALID",
                    data_format="NHWC",
                )
                with self.test_session() as sess:
                    outputs = np.squeeze(sess.run(outputs))
                pos_inp = np.squeeze(np.nonzero(inputs))
                pos_out = np.squeeze(np.nonzero(outputs))
                padding = padding_ops.same_padding_for_kernel(kshape, True)
                self.assertEqual(padding[0][0], pos_inp - pos_out)

    def test_same_padding_conv(self):
        for ishape in [[10], [11]]:
            inputs = np.zeros(ishape, dtype=np.float32)
            inputs[len(inputs) // 2] = 1
            for kshape in [[4], [5]]:
                kernel = np.zeros(kshape, dtype=np.float32)
                kernel[len(kernel) // 2] = 1
                outputs = tf.nn.conv2d_transpose(
                    tf.reshape(inputs, (1, 1, -1, 1)),
                    tf.reshape(kernel, (1, -1, 1, 1)),
                    (1, 1, ishape[0] + kshape[0] - 1, 1),
                    strides=(1, 1, 1, 1),
                    padding="VALID",
                    data_format="NHWC",
                )
                outputs = outputs[:, :, (kshape[0] - 1) : -(kshape[0] - 1), :]
                with self.test_session() as sess:
                    outputs = np.squeeze(sess.run(outputs))
                pos_inp = np.squeeze(np.nonzero(inputs))
                pos_out = np.squeeze(np.nonzero(outputs))
                padding = padding_ops.same_padding_for_kernel(kshape, False)
                self.assertEqual(padding[0][0], pos_inp - pos_out)


if __name__ == "__main__":
    tf.test.main()
