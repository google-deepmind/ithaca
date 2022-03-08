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
"""Ithaca model."""

from . import bigbird
from . import common_layers

import flax.linen as nn
import jax
import jax.numpy as jnp


class Model(nn.Module):
  """Transformer Model for sequence tagging."""
  vocab_char_size: int = 164
  vocab_word_size: int = 100004
  output_subregions: int = 85
  output_date: int = 160
  output_date_dist: bool = True
  output_return_emb: bool = False
  use_output_mlp: bool = True
  num_heads: int = 8
  num_layers: int = 6
  word_char_emb_dim: int = 192
  emb_dim: int = 512
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 1024
  causal_mask: bool = False
  feature_combine_type: str = 'concat'
  posemb_combine_type: str = 'add'
  region_date_pooling: str = 'first'
  learn_pos_emb: bool = True
  use_bfloat16: bool = False
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  activation_fn: str = 'gelu'
  model_type: str = 'bigbird'

  def setup(self):
    self.text_char_emb = nn.Embed(
        num_embeddings=self.vocab_char_size,
        features=self.word_char_emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0),
        name='char_embeddings')
    self.text_word_emb = nn.Embed(
        num_embeddings=self.vocab_word_size,
        features=self.word_char_emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0),
        name='word_embeddings')

  @nn.compact
  def __call__(self,
               text_char=None,
               text_word=None,
               text_char_onehot=None,
               text_word_onehot=None,
               text_char_emb=None,
               text_word_emb=None,
               padding=None,
               is_training=True):
    """Applies Ithaca model on the inputs."""

    if text_char is not None and padding is None:
      padding = jnp.where(text_char > 0, 1, 0)
    elif text_char_onehot is not None and padding is None:
      padding = jnp.where(text_char_onehot.argmax(-1) > 0, 1, 0)
    padding_mask = padding[..., jnp.newaxis]
    text_len = jnp.sum(padding, 1)

    if self.posemb_combine_type == 'add':
      posemb_dim = None
    elif self.posemb_combine_type == 'concat':
      posemb_dim = self.word_char_emb_dim
    else:
      raise ValueError('Wrong feature_combine_type value.')

    # Character embeddings
    if text_char is not None:
      x = self.text_char_emb(text_char)
    elif text_char_onehot is not None:
      x = self.text_char_emb.attend(text_char_onehot)
    elif text_char_emb is not None:
      x = text_char_emb
    else:
      raise ValueError('Wrong inputs.')

    # Word embeddings
    if text_word is not None:
      text_word_emb_x = self.text_word_emb(text_word)
    elif text_word_onehot is not None:
      text_word_emb_x = self.text_word_emb.attend(text_word_onehot)
    elif text_word_emb is not None:
      text_word_emb_x = text_word_emb
    else:
      raise ValueError('Wrong inputs.')

    if self.feature_combine_type == 'add':
      x = x + text_word_emb_x
    elif self.feature_combine_type == 'concat':
      x = jax.lax.concatenate([x, text_word_emb_x], 2)
    else:
      raise ValueError('Wrong feature_combine_type value.')

    # Positional embeddings
    pe_init = common_layers.sinusoidal_init(
        max_len=self.max_len) if self.learn_pos_emb else None
    x = common_layers.AddPositionEmbs(
        posemb_dim=posemb_dim,
        posemb_init=pe_init,
        max_len=self.max_len,
        combine_type=self.posemb_combine_type,
        name='posembed_input',
    )(
        x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)

    # Set floating point
    if self.use_bfloat16:
      x = x.astype(jnp.bfloat16)
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    if self.model_type == 'bigbird':
      model_block = bigbird.BigBirdBlock
    else:
      raise ValueError('Wrong model type specified.')

    for lyr in range(self.num_layers):
      x = model_block(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dtype=dtype,
          causal_mask=self.causal_mask,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          deterministic=not is_training,
          activation_fn=self.activation_fn,
          connectivity_seed=lyr,
          name=f'encoderblock_{lyr}',
      )(
          x,
          padding_mask=padding_mask,
      )
    x = common_layers.LayerNorm(dtype=dtype, name='encoder_norm')(x)
    torso_output = x

    # Bert logits
    if self.use_output_mlp:
      x_mask = common_layers.MlpBlock(
          out_dim=self.word_char_emb_dim,
          mlp_dim=self.emb_dim,
          dtype=dtype,
          out_dropout=False,
          dropout_rate=self.dropout_rate,
          deterministic=not is_training,
          activation_fn=self.activation_fn)(
              x)
    else:
      x_mask = nn.Dense(self.word_char_emb_dim)(x)

    char_embeddings = self.text_char_emb.embedding
    char_embeddings = nn.Dropout(rate=self.dropout_rate)(
        char_embeddings, deterministic=not is_training)
    logits_mask = jnp.matmul(x_mask, jnp.transpose(char_embeddings))

    # Next sentence prediction
    if self.use_output_mlp:
      logits_nsp = common_layers.MlpBlock(
          out_dim=2,
          mlp_dim=self.emb_dim,
          dtype=dtype,
          out_dropout=False,
          dropout_rate=self.dropout_rate,
          deterministic=not is_training,
          activation_fn=self.activation_fn)(
              x)
    else:
      logits_nsp = nn.Dense(2)(x)

    # Average over temporal dimension
    if self.region_date_pooling == 'average':
      x = jnp.multiply(padding_mask.astype(jnp.float32), x)
      x = jnp.sum(x, 1) / text_len.astype(jnp.float32)[..., None]
    elif self.region_date_pooling == 'sum':
      x = jnp.multiply(padding_mask.astype(jnp.float32), x)
      x = jnp.sum(x, 1)
    elif self.region_date_pooling == 'first':
      x = x[:, 0, :]
    else:
      raise ValueError('Wrong pooling type specified.')

    # Date pred
    if self.output_date_dist:
      output_date_dim = self.output_date
    else:
      output_date_dim = 1

    if self.use_output_mlp:
      pred_date = common_layers.MlpBlock(
          out_dim=output_date_dim,
          mlp_dim=self.emb_dim,
          dtype=dtype,
          out_dropout=False,
          dropout_rate=self.dropout_rate,
          deterministic=not is_training,
          activation_fn=self.activation_fn)(
              x)
    else:
      pred_date = nn.Dense(output_date_dim)(x)

    # Region logits
    if self.use_output_mlp:
      logits_subregion = common_layers.MlpBlock(
          out_dim=self.output_subregions,
          mlp_dim=self.emb_dim,
          dtype=dtype,
          out_dropout=False,
          dropout_rate=self.dropout_rate,
          deterministic=not is_training,
          activation_fn=self.activation_fn)(
              x)
    else:
      logits_subregion = nn.Dense(self.output_subregions)(x)

    outputs = (pred_date, logits_subregion, logits_mask, logits_nsp)
    if self.output_return_emb:
      return outputs, torso_output
    else:
      return outputs
