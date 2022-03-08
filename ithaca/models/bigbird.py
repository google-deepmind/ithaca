# Copyright 2021 the Ithaca Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformer using BigBird (https://arxiv.org/abs/2007.14062).

This implementation is from the Long Range Arena:
https://github.com/google-research/long-range-arena/tree/main/lra_benchmarks/models/bigbird
"""

from typing import Any, Optional

from . import bigbird_attention
from . import common_layers

from flax import linen as nn
import jax.numpy as jnp

_DEFAULT_BLOCK_SIZE = 64
_DEFAULT_NUM_RAND_BLOCKS = 3


class BigBirdBlock(nn.Module):
  """BigBird layer (https://arxiv.org/abs/2007.14062).

  Attributes:
    qkv_dim: dimension of the query/key/value
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: number of heads
    dtype: the dtype of the computation (default: float32).
    causal_mask: bool, mask future or not
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
    deterministic: bool, deterministic or not (to apply dropout)
    activation_fn: Activation function ("relu", "gelu")
    block_size: Size of attention blocks.
    num_rand_blocks: Number of random blocks.
    connectivity_seed: Optional seed for random block sparse attention.
  """

  qkv_dim: Any
  mlp_dim: Any
  num_heads: Any
  dtype: Any = jnp.float32
  causal_mask: bool = False
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  activation_fn: str = 'relu'
  block_size: int = _DEFAULT_BLOCK_SIZE
  num_rand_blocks: int = _DEFAULT_NUM_RAND_BLOCKS
  connectivity_seed: Optional[int] = None

  @nn.compact
  def __call__(self, inputs, inputs_segmentation=None, padding_mask=None):
    """Applies BigBirdBlock module.

    Args:
      inputs: input data
      inputs_segmentation: input segmentation info for packed examples.
      padding_mask: bool, mask padding tokens, [b, l, 1]

    Returns:
      output after transformer block.

    """

    # Attention block.
    assert inputs.ndim == 3
    x = common_layers.LayerNorm(dtype=self.dtype)(inputs)
    x = bigbird_attention.BigBirdSelfAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        deterministic=self.deterministic,
        block_size=self.block_size,
        num_rand_blocks=self.num_rand_blocks,
        connectivity_seed=self.connectivity_seed)(
            x,
            segmentation=inputs_segmentation,
            padding_mask=padding_mask,
        )
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=self.deterministic)
    x = x + inputs

    # MLP block.
    y = common_layers.LayerNorm(dtype=self.dtype)(x)
    y = common_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        deterministic=self.deterministic,
        activation_fn=self.activation_fn)(
            y)

    return x + y
