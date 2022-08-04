"""Padding ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports


def same_padding_for_kernel(shape, corr, strides_up=None):
    """Determine correct amount of padding for `same` convolution.

    To implement `'same'` convolutions, we first pad the image, and then perform a
    `'valid'` convolution or correlation. Given the kernel shape, this function
    determines the correct amount of padding so that the output of the convolution
    or correlation is the same size as the pre-padded input.

    Args:
      shape: Shape of the convolution kernel (without the channel dimensions).
      corr: Boolean. If `True`, assume cross correlation, if `False`, convolution.
      strides_up: If this is used for an upsampled convolution, specify the
        strides here. (For downsampled convolutions, specify `(1, 1)`: in that
        case, the strides don't matter.)

    Returns:
      The amount of padding at the beginning and end for each dimension.
    """
    rank = len(shape)
    if strides_up is None:
        strides_up = rank * (1,)

    if corr:
        padding = [(s // 2, (s - 1) // 2) for s in shape]
    else:
        padding = [((s - 1) // 2, s // 2) for s in shape]

    padding = [
        (
            (padding[i][0] - 1) // strides_up[i] + 1,
            (padding[i][1] - 1) // strides_up[i] + 1,
        )
        for i in range(rank)
    ]
    return padding
