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
"""Common layers used in models.

This implementation is from the Long Range Arena:
https://github.com/google-research/long-range-arena/tree/main/lra_benchmarks/models/bigbird
"""

# pylint: disable=attribute-defined-outside-init,g-bare-generic
from typing import Any, Callable, Iterable, Optional

from flax import linen as nn
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp
import numpy as np

PRNGKey = Any
Array = Any
Shape = Iterable[int]
Dtype = Any  # this could be a real type?

ACTIVATION_FN_DICT = {
    'relu': nn.relu,
    'gelu': nn.gelu,
}


def grid_restack(all_vecs):
  """Grid restack for meta-performer.

  Given multiple sequences (lists) of batch x len x dim,
  reshape this such that all positions are side by side.

  for example (for illustrative purposes):

  inputs: [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]
  outputs: [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

  Args:
    all_vecs: list of sequences of batch x len x dim

  Returns:
    Array of batch x (length x num_items) x dim.
  """
  cat_output = []
  for pos in range(all_vecs[0].shape[1]):
    pos_vecs = [x[:, None, pos, :] for x in all_vecs]
    cat_output += pos_vecs
  x2 = jnp.concatenate(cat_output, 1)
  return x2


def shift_right(x):
  """Shift the input to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[1] = (1, 0)  # Padding on axis=1
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return padded[:, :-1]


class Embed(nn.Module):
  """Embedding Module.

  A parameterized function from integers [0, n) to d-dimensional vectors.

  Attributes:
    mode: either 'input' or 'output' -> to share input/output embedding
    emb_init: embedding initializer
  """

  mode: str = 'input'
  emb_init: Callable = nn.initializers.normal(stddev=1.0)

  @nn.compact
  def __call__(self, inputs, num_embeddings, features):
    """Applies Embed module.

    Args:
      inputs: input data
      num_embeddings: number of embedding
      features: size of the embedding dimension

    Returns:
      output which is embedded input data
    """
    embedding = self.param('embedding', self.emb_init,
                           (num_embeddings, features))
    if self.mode == 'input':
      if inputs.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
        raise ValueError('Input type must be an integer or unsigned integer.')
      return jnp.take(embedding, inputs, axis=0)
    if self.mode == 'output':
      return jnp.einsum('bld,vd->blv', inputs, embedding)


def sinusoidal_init(max_len=2048, replicate_tf=False):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input
      replicate_tf: replicate TF periodic encoding exactly

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    if replicate_tf:
      half_d_feature = d_feature // 2
      div_term = np.exp(
          np.arange(half_d_feature) * -(np.log(10000.0) / (half_d_feature - 1)))
      pe[:, :half_d_feature] = np.sin(position * div_term)
      pe[:, half_d_feature:] = np.cos(position * div_term)
    else:
      div_term = np.exp(
          np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
      pe[:, 0::2] = np.sin(position * div_term)
      pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    posemb_init: positional embedding initializer, if None, then use a fixed
      (non-learned) sinusoidal embedding table.
    max_len: maximum possible length for the input.
    replicate_original: replicate original periodic encoding exactly
  """

  posemb_init: Optional[Callable] = None
  posemb_dim: Optional[int] = None
  max_len: int = 512
  combine_type: str = 'concat'
  replicate_tf: bool = False

  @nn.compact
  def __call__(self, inputs, inputs_positions=None, cache=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.
      cache: flax attention cache for fast decoding.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    batch_size = inputs.shape[0]
    length = inputs.shape[1]
    if self.posemb_dim is None or self.combine_type == 'add':
      self.posemb_dim = inputs.shape[-1]
    pos_emb_shape = (1, self.max_len, self.posemb_dim)
    if self.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(
          max_len=self.max_len,
          replicate_tf=self.replicate_tf,
      )(None, pos_emb_shape, None)
    else:
      pos_embedding = self.param('pos_embedding', self.posemb_init,
                                 pos_emb_shape)
    pe = pos_embedding[:, :length, :]
    # We abuse the same attention Cache mechanism to run positional embeddings
    # in fast predict mode. We could use state variables instead, but this
    # simplifies invocation with a single top-level cache context manager.
    # We only use the cache's position index for tracking decoding position.
    if cache:
      if self.is_initializing():
        cache.store(np.array((4, 1, 1), dtype=np.int32))
      else:
        cache_entry = cache.retrieve(None)
        i = cache_entry.i
        cache.store(cache_entry.replace(i=cache_entry.i + 1))
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding, jnp.array((0, i, 0)),
                               jnp.array((1, 1, df)))
    if inputs_positions is None:
      # normal unpacked case:
      if self.combine_type == 'add':
        return inputs + pe
      elif self.combine_type == 'concat':
        pe_broadcast = np.repeat(pe, batch_size, axis=0)
        return lax.concatenate([inputs, pe_broadcast], 2)
      else:
        raise ValueError('Wrong type value.')
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP block."""

  mlp_dim: int
  dtype: Any = jnp.float32
  out_dim: Optional[int] = None
  out_dropout: bool = True
  dropout_rate: float = 0.1
  deterministic: bool = False
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  activation_fn: str = 'gelu'

  @nn.compact
  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(
            inputs)
    x = ACTIVATION_FN_DICT[self.activation_fn](x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=self.deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(
            x)
    if self.out_dropout:
      output = nn.Dropout(rate=self.dropout_rate)(
          output, deterministic=self.deterministic)
    return output


def classifier_head(encoded, num_classes, mlp_dim, pooling_mode='MEAN'):
  """Classifier head.

  We put this here just so that all models consistently call the same function.

  Args:
    encoded: tensor inputs are shape of [bs, len, dim].
    num_classes: int, number of classes
    mlp_dim: int, dim of intermediate MLP.
    pooling_mode: str, string dictating pooling op {MEAN}

  Returns:
    tensor of shape [bs, num_classes]

  """
  if pooling_mode == 'MEAN':
    encoded = jnp.mean(encoded, axis=1)
  elif pooling_mode == 'SUM':
    encoded = jnp.sum(encoded, axis=1)
  elif pooling_mode == 'FLATTEN':
    encoded = encoded.reshape((encoded.shape[0], -1))
  elif pooling_mode == 'CLS':
    encoded = encoded[:, 0]
  else:
    raise NotImplementedError('Pooling not supported yet.')
  encoded = nn.Dense(mlp_dim, name='mlp')(encoded)
  encoded = nn.relu(encoded)
  encoded = nn.Dense(num_classes, name='logits')(encoded)
  return encoded


class LayerNorm(nn.Module):
  """Layer Norm to replicate tf.contrib."""
  epsilon: Optional[float] = None
  dtype: Any = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones

  @nn.compact
  def __call__(self, x):
    if self.epsilon is None:
      epsilon = 1e-12 if self.dtype != jnp.float16 else 1e-3
    else:
      epsilon = self.epsilon
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean = jnp.mean(x, axis=-1, keepdims=True)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    var = mean2 - lax.square(mean)
    mul = lax.rsqrt(var + epsilon)
    if self.use_scale:
      mul = mul * jnp.asarray(
          self.param('scale', self.scale_init, (features,)), self.dtype)
    y = x * mul
    if self.use_bias:
      y = y + jnp.asarray(
          self.param('bias', self.bias_init, (features,)), self.dtype)
    y -= mean * mul
    return jnp.asarray(y, self.dtype)
