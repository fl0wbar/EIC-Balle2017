"""Initializers for layer classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops


class IdentityInitializer(object):
    """Initialize to the identity kernel with the given shape.

    This creates an n-D kernel suitable for `SignalConv*` with the requested
    support that produces an output identical to its input (except possibly at the
    signal boundaries).

    Note: The identity initializer in `tf.initializers` is only suitable for
    matrices, not for n-D convolution kernels (i.e., no spatial support).
    """

    def __init__(self, gain=1):
        self.gain = float(gain)

    def __call__(self, shape, dtype=None, partition_info=None):
        del partition_info  # unused
        assert len(shape) > 2, shape

        support = tuple(shape[:-2]) + (1, 1)
        indices = [[s // 2 for s in support]]
        updates = array_ops.constant([self.gain], dtype=dtype)
        kernel = array_ops.scatter_nd(indices, updates, support)

        assert shape[-2] == shape[-1], shape
        if shape[-1] != 1:
            kernel *= linalg_ops.eye(shape[-1], dtype=dtype)

        return kernel
