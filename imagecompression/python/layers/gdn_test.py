"""Tests of GDN layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow as tf
import imagecompression.python.layers.gdn as gdn


class GDNTest(tf.test.TestCase):
    def _run_gdn(self, x, shape, inverse, rectify, data_format):
        inputs = tf.placeholder(tf.float32, shape)
        layer = gdn.GDN(inverse=inverse, rectify=rectify, data_format=data_format)
        outputs = layer(inputs)
        with self.test_session() as sess:
            tf.global_variables_initializer().run()
            (y,) = sess.run([outputs], {inputs: x})
        return y

    def test_invalid_data_format(self):
        x = np.random.uniform(size=(1, 2, 3, 4))
        with self.assertRaises(ValueError):
            self._run_gdn(x, x.shape, False, False, "NHWC")

    def test_unknown_dim(self):
        x = np.random.uniform(size=(1, 2, 3, 4))
        with self.assertRaises(ValueError):
            self._run_gdn(x, 4 * [None], False, False, "channels_last")

    def test_channels_last(self):
        for ndim in [2, 3, 4, 5, 6]:
            x = np.random.uniform(size=(1, 2, 3, 4, 5, 6)[:ndim])
            y = self._run_gdn(x, x.shape, False, False, "channels_last")
            self.assertEqual(x.shape, y.shape)
            self.assertAllClose(y, x / np.sqrt(1 + 0.1 * (x**2)), rtol=0, atol=1e-6)

    def test_channels_first(self):
        for ndim in [2, 3, 4, 5, 6]:
            x = np.random.uniform(size=(6, 5, 4, 3, 2, 1)[:ndim])
            y = self._run_gdn(x, x.shape, False, False, "channels_first")
            self.assertEqual(x.shape, y.shape)
            self.assertAllClose(y, x / np.sqrt(1 + 0.1 * (x**2)), rtol=0, atol=1e-6)

    def test_wrong_dims(self):
        x = np.random.uniform(size=(3,))
        with self.assertRaises(ValueError):
            self._run_gdn(x, x.shape, False, False, "channels_last")
        with self.assertRaises(ValueError):
            self._run_gdn(x, x.shape, True, True, "channels_first")

    def test_igdn(self):
        x = np.random.uniform(size=(1, 2, 3, 4))
        y = self._run_gdn(x, x.shape, True, False, "channels_last")
        self.assertEqual(x.shape, y.shape)
        self.assertAllClose(y, x * np.sqrt(1 + 0.1 * (x**2)), rtol=0, atol=1e-6)

    def test_rgdn(self):
        x = np.random.uniform(-0.5, 0.5, size=(1, 2, 3, 4))
        y = self._run_gdn(x, x.shape, False, True, "channels_last")
        self.assertEqual(x.shape, y.shape)
        x = np.maximum(x, 0)
        self.assertAllClose(y, x / np.sqrt(1 + 0.1 * (x**2)), rtol=0, atol=1e-6)


if __name__ == "__main__":
    tf.test.main()
