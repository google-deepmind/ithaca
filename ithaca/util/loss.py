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
"""Loss functions."""
import chex
from flax.deprecated import nn
import jax
import jax.numpy as jnp


def smooth_labels(labels, num_classes, label_smoothing):
  if not 0 <= label_smoothing < 1:
    raise ValueError(
        f"'label_smoothing is {label_smoothing} and should be in [0, 1)")
  one = jax.lax.convert_element_type(1, labels.dtype)
  label_smoothing = jax.lax.convert_element_type(label_smoothing,
                                                 labels.dtype)
  num_classes = jax.lax.convert_element_type(num_classes, labels.dtype)
  return (one - label_smoothing) * labels + (label_smoothing / num_classes)


def categorical_kl_divergence(p_logits, q_logits, temperature=1.):
  """Compute the KL between two categorical distributions from their logits.

  Args:
    p_logits: unnormalized logits for the first distribution.
    q_logits: unnormalized logits for the second distribution.
    temperature: the temperature for the softmax distribution, defaults at 1.

  Returns:
    the kl divergence between the distributions.
  """
  chex.assert_type([p_logits, q_logits], float)

  p_logits /= temperature
  q_logits /= temperature

  p = jax.nn.softmax(p_logits)
  log_p = jax.nn.log_softmax(p_logits)
  log_q = jax.nn.log_softmax(q_logits)
  kl = jnp.sum(p * (log_p - log_q), axis=-1)
  return jax.nn.relu(kl)  # Guard against numerical issues giving negative KL.


def cross_entropy_label_smoothing_loss(logits,
                                       labels,
                                       mask=None,
                                       label_smoothing=0.1):
  """Cross entropy loss with label smoothing."""

  num_classes = logits.shape[-1]
  labels_onehot = jax.nn.one_hot(labels, num_classes, dtype=logits.dtype)
  if label_smoothing > 0:
    labels_onehot = smooth_labels(labels_onehot, num_classes, label_smoothing)

  loss = -jnp.sum(labels_onehot * jax.nn.log_softmax(logits), axis=-1)
  if mask is not None:
    loss = jnp.multiply(loss, mask.astype(logits.dtype))
  return loss


@jax.vmap
def cross_entropy_loss(logits, label):
  logits = nn.log_softmax(logits)
  return -logits[label]


def cross_entropy_mask_loss(logits, label, mask):
  nll = -nn.log_softmax(logits)[label]
  loss = jnp.multiply(nll, mask.astype(logits.dtype))
  return loss


def date_loss_l2(pred,
                 target_min,
                 target_max,
                 mask):
  """L2 loss function for dates."""
  pred = jnp.squeeze(pred, 0)

  loss = 0.
  loss += (pred - target_min)**2 * jnp.less(pred, target_min).astype(
      pred.dtype)
  loss += (pred - target_max)**2 * jnp.greater(pred, target_max).astype(
      pred.dtype)

  # Mask loss
  loss = jnp.multiply(loss, mask.astype(loss.dtype))
  return loss


def date_loss_l1(pred,
                 target_min,
                 target_max,
                 mask):
  """L1 loss function for dates."""
  pred = jnp.squeeze(pred, 0)

  loss = 0.
  loss += jnp.abs(pred - target_min) * jnp.less(pred, target_min).astype(
      pred.dtype)
  loss += jnp.abs(pred - target_max) * jnp.greater(pred, target_max).astype(
      pred.dtype)

  # Mask loss
  loss = jnp.multiply(loss, mask.astype(loss.dtype))
  return loss
