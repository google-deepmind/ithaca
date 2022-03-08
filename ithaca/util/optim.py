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
"""Optimizer utilities."""

from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union
import jax
import jax.numpy as jnp


def linear_weight(global_step, start, end):
  """Linear weight increase."""
  if end <= 0:
    return 1.
  t = jnp.maximum(0., global_step - start)
  w = t / (end - start)
  w = jnp.minimum(w, 1.)
  return w


def linear_warmup_and_sqrt_decay(global_step, max_lr, warmup_steps):
  """Linear warmup and then an inverse square root decay of learning rate."""
  linear_ratio = max_lr / warmup_steps
  decay_ratio = jnp.power(warmup_steps * 1.0, 0.5) * max_lr
  return jnp.minimum(linear_ratio * global_step,
                     decay_ratio * jnp.power(global_step, -0.5))


def create_learning_rate_scheduler(
    factors='constant * linear_warmup * rsqrt_decay',
    base_learning_rate=0.5,
    warmup_steps=1000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: string, factors separated by '*' that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: int, how many steps to warm up for in the warmup schedule.
    decay_factor: float, the amount to decay the learning rate by.
    steps_per_decay: int, how often to decay the learning rate.
    steps_per_cycle: int, steps per cycle when using cosine decay.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= base_learning_rate
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'decay_every':
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == 'cosine_decay':
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError('Unknown factor %s.' % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


# pylint:disable=no-value-for-parameter
OptState = NamedTuple  # Transformation states are (possibly empty) namedtuples.
Params = Any  # Parameters are arbitrary nests of `jnp.ndarrays`.
Updates = Params  # Gradient updates are of the same type as parameters.


class GradientTransformation(NamedTuple):
  """Optax transformations consists of a function pair: (initialise, update)."""
  init: Callable[  # Function used to initialise the transformation's state.
      [Params], Union[OptState, Sequence[OptState]]]
  update: Callable[  # Function used to apply a transformation.
      [Updates, OptState, Optional[Params]], Tuple[Updates, OptState]]


class ClipByGlobalNormState(OptState):
  """The `clip_by_global_norm` transformation is stateless."""


def unitwise_norm(x):
  """Computes norms of each output unit separately."""
  if len(jnp.squeeze(x).shape) <= 1:  # Scalars and vectors
    axis = None
    keepdims = False
  elif len(x.shape) in [2, 3]:  # Linear layers of shape IO or multihead linear
    axis = 0
    keepdims = True
  elif len(x.shape) == 4:  # Conv kernels of shape HWIO
    axis = [0, 1, 2,]
    keepdims = True
  else:
    raise ValueError(f'Got a parameter with shape not in [1, 2, 3, 4]! {x}')
  return jnp.sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5


def unitwise_clip(g_norm, max_norm, grad):
  """Applies gradient clipping unit-wise."""
  trigger = g_norm < max_norm
  # This little max(., 1e-6) is distinct from the normal eps and just prevents
  # division by zero. It technically should be impossible to engage.
  clipped_grad = grad * (max_norm / jnp.maximum(g_norm, 1e-6))
  return jnp.where(trigger, grad, clipped_grad)


def adaptive_grad_clip(clipping, eps=1e-3) -> GradientTransformation:
  """Clip updates to be at most clipping * parameter_norm, unit-wise.

  References:
    [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
    Recognition Without Normalization. (https://arxiv.org/abs/2102.06171)

  Args:
    clipping: Maximum allowed ratio of update norm to parameter norm.
    eps: epsilon term to prevent clipping of zero-initialized params.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return ClipByGlobalNormState()

  def update_fn(updates, state, params):
    g_norm = jax.tree_map(unitwise_norm, updates)
    p_norm = jax.tree_map(unitwise_norm, params)
    # Maximum allowable norm
    max_norm = jax.tree_map(lambda x: clipping * jnp.maximum(x, eps), p_norm)
    # If grad norm > clipping * param_norm, rescale
    updates = jax.tree_multimap(unitwise_clip, g_norm, max_norm, updates)
    return updates, state

  return GradientTransformation(init_fn, update_fn)
