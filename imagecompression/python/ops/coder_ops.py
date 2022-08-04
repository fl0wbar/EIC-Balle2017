"""Range coder operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensorflow.contrib.coder.python.ops import coder_ops

pmf_to_quantized_cdf = coder_ops.pmf_to_quantized_cdf
range_decode = coder_ops.range_decode
range_encode = coder_ops.range_encode
