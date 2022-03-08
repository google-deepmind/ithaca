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
"""Ithaca: Restoring and attributing ancient texts with deep neural networks."""

import bz2
import distutils
import functools
import glob
import os
import pickle

from absl import app
from absl import flags
from absl import logging
import dataloader
from ithaca.models.model import Model
from ithaca.util.alphabet import GreekAlphabet
from ithaca.util.loss import categorical_kl_divergence
from ithaca.util.loss import cross_entropy_label_smoothing_loss
from ithaca.util.loss import cross_entropy_loss
from ithaca.util.loss import cross_entropy_mask_loss
from ithaca.util.loss import date_loss_l1
from ithaca.util.optim import adaptive_grad_clip
from ithaca.util.optim import linear_warmup_and_sqrt_decay
from ithaca.util.optim import linear_weight
from ithaca.util.region_names import load_region_maps
import jax
import jax.numpy as jnp
from jaxline import experiment
from jaxline import platform
from jaxline import utils as jl_utils
import numpy as np
import optax
import tensorflow_datasets.public_api as tfds

FLAGS = flags.FLAGS


class Experiment(experiment.AbstractExperiment):
  """Ithaca experiment."""

  # Holds a map from object properties that will be checkpointed to their name
  # within a checkpoint. Currently it is assume that these are all sharded
  # device arrays.
  CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_opt_state': 'opt_state',
  }

  def __init__(self, mode, init_rng, config):
    """Initializes experiment."""

    super(Experiment, self).__init__(mode=mode)
    self.mode = mode
    self.init_rng = init_rng
    self.config = config

    # Same random key on each device.
    self._rng_key = jl_utils.bcast_local_devices(self.init_rng)

    # Checkpointed experiment state.
    self._params = None
    self._opt_state = None

    # Input pipelines.
    self._train_input = None
    self._eval_input = None

    # Forward and update functions.
    self.forward = Model(**self.config.model)
    self._update_func = jax.pmap(
        self._update_func, axis_name='i', donate_argnums=(0, 1))

    self._learning_rate_fn = functools.partial(
        linear_warmup_and_sqrt_decay,
        max_lr=self.config.optimizer.kwargs.learning_rate,
        warmup_steps=self.config.optimizer.warmup)

    self._opt_init, self._opt_update = self.optimizer()

    if 'use_jit' in self.config.evaluation and self.config.evaluation.use_jit:
      self._eval_batch = jax.jit(self._eval_batch)

    # Create alphabet
    alphabet_kwargs = dict(self.config.alphabet)
    wordlist_path = alphabet_kwargs.pop('wordlist_path')
    with open(wordlist_path, 'r') as f:
      self._alphabet = GreekAlphabet(wordlist_file=f, **alphabet_kwargs)

    # Create region mapping
    self._region_map = {'main': None, 'sub': None}
    if self.config.dataset.region_main_path:
      with open(self.config.dataset.region_main_path, 'r') as f:
        self._region_map['main'] = load_region_maps(f)
    if self.config.dataset.region_sub_path:
      with open(self.config.dataset.region_sub_path, 'r') as f:
        self._region_map['sub'] = load_region_maps(f)

  def optimizer(self):
    config_opt = self.config.optimizer

    kwargs = config_opt.kwargs.to_dict()
    kwargs['learning_rate'] = self._learning_rate_fn
    opt = getattr(optax, config_opt.name)(**kwargs)

    if hasattr(config_opt, 'clip_adaptive') and config_opt.clip_adaptive:
      if config_opt.clip_level > 0.:
        opt = optax.chain(adaptive_grad_clip(config_opt.clip_level), opt)
    elif config_opt.clip_level > 0.:
      opt = optax.chain(optax.clip_by_global_norm(config_opt.clip_level), opt)
    return opt

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def step(self, global_step, rng, **unused_args):
    """See base class."""

    if self._train_input is None:
      self._initialize_train(rng)

    batch = next(self._train_input)
    (self._params, self._opt_state, scalars) = (
        self._update_func(self._params, self._opt_state, global_step, batch,
                          rng))

    scalars = jl_utils.get_first(scalars)
    return scalars

  def _initialize_train(self, rng):
    # Check we haven't already restored params
    if self._params is None:
      logging.info(
          'Initializing parameters rather than restoring from checkpoint.')
      batch = next(self._build_train_input())

      rng = jl_utils.get_first(rng)
      params_rng, dropout_rng = jax.random.split(rng)
      params_rng = jl_utils.bcast_local_devices(params_rng)
      dropout_rng = jl_utils.bcast_local_devices(dropout_rng)
      init_net = jax.pmap(
          functools.partial(self.forward.init, is_training=True))
      self._params = init_net({
          'params': params_rng,
          'dropout': dropout_rng
      },
                              text_char=batch['text_char'],
                              text_word=batch['text_word'])

      init_opt = jax.pmap(self._opt_init)
      self._opt_state = init_opt(self._params)

      self._train_input = jl_utils.py_prefetch(self._build_train_input)
      self._train_input = jl_utils.double_buffer_on_gpu(self._train_input)

  def _build_train_input(self):
    """See base class."""
    num_devices = jax.device_count()
    global_batch_size = self.config.training.batch_size
    per_device_batch_size, ragged = divmod(global_batch_size, num_devices)
    logging.info(
        'num_devices: %d, per_device_batch_size: %d, global_batch_size: %d',
        num_devices, per_device_batch_size, global_batch_size)

    if ragged:
      raise ValueError(
          f'Global batch size {global_batch_size} must be divisible by '
          f'num devices {num_devices}')

    config_dataset = self.config.dataset
    with open(config_dataset.dataset_path) as dataset_file:
      ds = dataloader.loader_tf(
          per_device_batch_size,
          config_dataset,
          self._region_map,
          alphabet=self._alphabet,
          dataset_file=dataset_file,
          mode='train')

    ds = ds.batch(jax.local_device_count())
    return iter(tfds.as_numpy(ds))

  def _loss_fn(self, params, batch, global_step, rng):
    text_char = batch['text_char']
    text_word = batch['text_word']
    text_unmasked = batch['text_unmasked']
    text_mask = batch['text_mask']
    next_sentence_mask = batch['next_sentence_mask']
    next_sentence_label = batch['next_sentence_label']
    subregion = batch['region_sub_id']
    date_min = batch['date_min']
    date_max = batch['date_max']
    date_dist = batch['date_dist']
    date_available = batch['date_available']
    eps = 1e-6

    (date_pred, subregion_logits, mask_logits, nsp_logits) = self.forward.apply(
        params,
        text_char=text_char,
        text_word=text_word,
        text_char_onehot=None,
        text_word_onehot=None,
        is_training=True,
        rngs={'dropout': rng})

    date_loss = 0.
    subregion_loss = 0.
    subregion_accuracy = 0.
    mask_loss = 0.
    mask_accuracy = 0.
    nsp_loss = 0.
    nsp_accuracy = 0.

    # Date loss
    if self.config.loss.date.enabled:
      if self.config.loss.date.label_smoothing > 0:
        date_dist_prob = jnp.exp(date_dist)  # logprob to prob
        date_dist_prob_smooth = date_dist_prob * jax.random.uniform(
            rng,
            shape=date_dist_prob.shape,
            dtype=date_dist_prob.dtype,
            minval=1 - self.config.loss.date.label_smoothing,
            maxval=1 + self.config.loss.date.label_smoothing)
        date_dist_prob_smooth /= date_dist_prob_smooth.sum(axis=-1)[:,
                                                                    jnp.newaxis]

        date_dist_prob_smooth = jnp.clip(date_dist_prob_smooth, 1e-6, 1)
        date_dist = jnp.log(date_dist_prob_smooth)

      date_loss = 0.
      if 'l1' in self.config.loss.date.type.split('+'):
        date_pred_x = jnp.arange(
            self.config.dataset.date_min +
            self.config.dataset.date_interval / 2,
            self.config.dataset.date_max +
            self.config.dataset.date_interval / 2,
            self.config.dataset.date_interval).reshape(-1, 1)
        date_pred_val = jnp.dot(jax.nn.softmax(date_pred, axis=-1), date_pred_x)
        date_loss_l1_ = jax.vmap(date_loss_l1)(date_pred_val, date_min,
                                               date_max, date_available)
        jnp.nan_to_num(date_loss_l1_, copy=False)
        date_loss += (
            jnp.mean(date_loss_l1_, axis=0) * self.config.loss.date.weight_l1)

      if 'dist' in self.config.loss.date.type.split('+'):
        date_loss_dist_ = categorical_kl_divergence(date_dist, date_pred)
        date_loss_dist_ *= date_available
        jnp.nan_to_num(date_loss_dist_, copy=False)
        date_loss += (
            jnp.mean(date_loss_dist_, axis=0) *
            self.config.loss.date.weight_dist)

      date_loss *= linear_weight(global_step, self.config.loss.date.step_start,
                                 self.config.loss.date.step_end)

    # Region and subregion loss
    if self.config.loss.region.enabled:
      subregion_loss = jnp.mean(
          cross_entropy_label_smoothing_loss(
              subregion_logits,
              subregion,
              label_smoothing=self.config.loss.region.label_smoothing), 0)
      jnp.nan_to_num(subregion_loss, copy=False)
      subregion_loss *= self.config.loss.region.weight
      subregion_accuracy = jnp.mean(
          jnp.argmax(subregion_logits, -1) == subregion)

      w = linear_weight(global_step, self.config.loss.region.step_start,
                        self.config.loss.region.step_end)
      subregion_loss *= w

    # Mask loss
    if self.config.loss.mask.enabled:
      mask_loss = jnp.sum(
          cross_entropy_label_smoothing_loss(
              mask_logits,
              text_unmasked,
              text_mask,
              label_smoothing=self.config.loss.mask.label_smoothing), 1)  # [B]
      assert mask_loss.ndim == 1
      jnp.nan_to_num(mask_loss, copy=False)
      mask_loss = jnp.mean(mask_loss, 0) * self.config.loss.mask.weight  # []
      mask_all_accuracy = (jnp.argmax(mask_logits, -1) == text_unmasked).astype(
          mask_logits.dtype)
      mask_accuracy = jnp.divide(
          jnp.sum(
              jnp.multiply(mask_all_accuracy,
                           text_mask.astype(mask_logits.dtype))),
          jnp.sum(text_mask) + eps)

      mask_loss *= linear_weight(global_step, self.config.loss.mask.step_start,
                                 self.config.loss.mask.step_end)

    # NSP loss
    if self.config.loss.nsp.enabled:
      nsp_loss = jnp.sum(
          jax.vmap(jax.vmap(cross_entropy_mask_loss))(nsp_logits,
                                                      next_sentence_label,
                                                      next_sentence_mask),
          1)  # [B]
      assert nsp_loss.ndim == 1
      jnp.nan_to_num(nsp_loss, copy=False)
      nsp_loss = jnp.mean(nsp_loss, 0) * self.config.loss.nsp.weight
      nsp_all_accuracy = (jnp.argmax(
          nsp_logits, -1) == next_sentence_label).astype(nsp_logits.dtype)
      nsp_accuracy = jnp.divide(
          jnp.sum(
              jnp.multiply(nsp_all_accuracy,
                           next_sentence_mask.astype(nsp_logits.dtype))),
          jnp.sum(next_sentence_mask) + eps)
      nsp_loss *= linear_weight(global_step, self.config.loss.nsp.step_start,
                                self.config.loss.nsp.step_end)

    loss = date_loss + subregion_loss + mask_loss + nsp_loss
    scaled_loss = loss / jax.device_count()
    # NOTE: We use scaled_loss for grads and unscaled for logging.
    return scaled_loss, (loss, date_loss, subregion_loss, subregion_accuracy,
                         mask_loss, mask_accuracy, nsp_loss, nsp_accuracy)

  def _update_func(self, params, opt_state, global_step, batch, rng):
    """Applies an update to parameters and returns new state."""
    # This function computes the gradient of the first output of loss_fn and
    # passes through the other arguments unchanged.
    grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)
    scaled_grads, (loss, date_loss, subregion_loss, subregion_accuracy,
                   mask_loss, mask_accuracy, nsp_loss,
                   nsp_accuracy) = grad_loss_fn(params, batch, global_step, rng)

    scaled_grads = jax.tree_map(jnp.nan_to_num, scaled_grads)
    grads = jl_utils.tree_psum(scaled_grads, axis_name='i')

    # Compute and apply updates via our optimizer.
    learning_rate = self._learning_rate_fn(global_step)
    updates, opt_state = self._opt_update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)

    # Scalars to log (note: we log the mean across all hosts/devices).
    scalars = {
        'loss/train': loss,
        'loss/date': date_loss,
        'loss/subregion': subregion_loss,
        'loss/mask': mask_loss,
        'loss/nsp': nsp_loss,
        'accuracy/subregion': subregion_accuracy,
        'accuracy/mask': mask_accuracy,
        'accuracy/nsp': nsp_accuracy,
        'opt/learning_rate': learning_rate,
        'opt/grad_norm': optax.global_norm(grads),
        'opt/param_norm': optax.global_norm(params),
    }
    scalars = jax.lax.pmean(scalars, axis_name='i')

    return params, opt_state, scalars

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(self, global_step, rng, **unused_kwargs):
    """See base class."""

    if self._eval_input is None:
      self._initialize_eval()

    global_step = np.array(jl_utils.get_first(global_step))
    summary, outputs = self._eval_epoch(jl_utils.get_first(rng))

    for k, v in summary.items():
      summary[k] = np.array(v)

    score = summary['score/eval']
    logging.info('[Step %d] eval_score=%.2f', global_step, score)

    # Log outputs
    checkpoint_dir = jl_utils.get_checkpoint_dir(FLAGS.config,
                                                 jax.process_index())
    outputs_path = os.path.join(checkpoint_dir, 'best_outputs.pkl.bz2')
    score_path = os.path.join(checkpoint_dir, 'best_score.txt')
    model_log_path = os.path.join(checkpoint_dir, 'model_log')
    best_model_log_path = os.path.join(checkpoint_dir, 'best_model_log')

    # Check for preexisting outputs
    best_score = None
    best_step = None
    if os.path.exists(score_path):
      with open(score_path, 'r') as f:
        tok = f.read().strip().split(' ')
        best_step = int(tok[0])
        best_score = float(tok[1])

    # Store outputs if score is better
    if best_score is None or (score > best_score and global_step > best_step):
      best_score = score

      with open(score_path, 'w') as f:
        f.write(f'{global_step} {best_score}')

      with open(outputs_path, 'wb') as f:
        outputs_pkl = pickle.dumps(outputs, protocol=2)
        outputs_pkl_bz2 = bz2.compress(outputs_pkl)
        f.write(outputs_pkl_bz2)

      if self.config.evaluation.store_model_log:
        if os.path.isdir(best_model_log_path):
          map(os.remove, glob.glob(best_model_log_path + '/*'))
        else:
          os.makedirs(best_model_log_path)
        distutils.dir_util.copy_tree(model_log_path, best_model_log_path)

      logging.info('[Step %d] Writing eval outputs: %s.', global_step,
                   outputs_path)

    # Log best score
    summary['score/eval_best'] = best_score

    return summary

  def _initialize_eval(self):
    self._eval_input = jl_utils.py_prefetch(self._build_eval_input)

  def _build_eval_input(self):
    """Builds the evaluation input pipeline."""
    config_dataset = self.config.dataset
    with open(config_dataset.dataset_path) as dataset_file:
      ds = dataloader.loader_tf(
          self.config.evaluation.batch_size,
          config_dataset,
          self._region_map,
          alphabet=self._alphabet,
          dataset_file=dataset_file,
          mode=self.config.evaluation.mode)

    return iter(tfds.as_numpy(ds))

  def _eval_batch(self, params, batch, rng):
    """Evaluates a batch."""
    phi_id = batch['id']
    text_char = batch['text_char']
    text_word = batch['text_word']
    text_unmasked = batch['text_unmasked']
    text_mask = batch['text_mask']
    next_sentence_mask = batch['next_sentence_mask']
    next_sentence_label = batch['next_sentence_label']
    subregion = batch['region_sub_id']
    date_min = batch['date_min']
    date_max = batch['date_max']
    date_dist = batch['date_dist']
    date_available = batch['date_available']

    # with hlogging.context() as log:
    (date_pred, subregion_logits, mask_logits, nsp_logits) = self.forward.apply(
        params,
        text_char=text_char,
        text_word=text_word,
        text_char_onehot=None,
        text_word_onehot=None,
        is_training=False,
        rngs={'dropout': rng})

    # Log model weights
    model_log = {}

    subregion_loss = 0.
    subregion_accuracy = 0.
    date_loss = 0.
    date_l1_loss = 0.
    nsp_loss = 0.
    nsp_accuracy = 0.
    # eps = 1e-6

    date_count = 0
    mask_count = 0
    nsp_count = 0

    # Date loss
    if self.config.loss.date.enabled:
      date_pred_x = jnp.arange(
          self.config.dataset.date_min + self.config.dataset.date_interval / 2,
          self.config.dataset.date_max + self.config.dataset.date_interval / 2,
          self.config.dataset.date_interval).reshape(-1, 1)
      date_pred_val = jnp.dot(jax.nn.softmax(date_pred, axis=-1), date_pred_x)
      date_l1_loss = jnp.sum(
          jax.vmap(date_loss_l1)(date_pred_val, date_min, date_max,
                                 date_available),
          axis=0)

      if 'l1' in self.config.loss.date.type.split('+'):
        date_loss += date_l1_loss * self.config.loss.date.weight_l1

      if 'dist' in self.config.loss.date.type.split('+'):
        date_loss_dist_ = categorical_kl_divergence(date_dist, date_pred)
        date_loss_dist_ *= date_available
        date_loss += (
            jnp.sum(date_loss_dist_, axis=0) *
            self.config.loss.date.weight_dist)

      date_count = jnp.sum(date_available)

    # Region and subregion loss
    if self.config.loss.region.enabled:
      subregion_loss = jnp.sum(
          cross_entropy_loss(subregion_logits, subregion), 0)
      subregion_loss *= self.config.loss.region.weight
      subregion_accuracy = jnp.mean(
          jnp.argmax(subregion_logits, -1) == subregion)

    # Mask loss
    if self.config.loss.mask.enabled:
      mask_loss = jnp.sum(
          cross_entropy_label_smoothing_loss(
              mask_logits, text_unmasked, text_mask, label_smoothing=0),
          1)  # [B]
      # mask_loss /= jnp.sum(text_mask, axis=1) + eps  # [B]
      assert mask_loss.ndim == 1
      mask_loss = jnp.mean(mask_loss, 0) * self.config.loss.mask.weight  # []

      mask_all_accuracy = (jnp.argmax(mask_logits, -1) == text_unmasked).astype(
          mask_logits.dtype)
      mask_accuracy = jnp.sum(
          jnp.multiply(mask_all_accuracy, text_mask.astype(mask_logits.dtype)))
      mask_count = jnp.sum(text_mask)

    # NSP loss
    if self.config.loss.nsp.enabled:
      nsp_loss = jnp.sum(
          jax.vmap(jax.vmap(cross_entropy_mask_loss))(nsp_logits,
                                                      next_sentence_label,
                                                      next_sentence_mask),
          1)  # [B]
      assert nsp_loss.ndim == 1
      nsp_loss = jnp.sum(nsp_loss, 0) * self.config.loss.nsp.weight
      nsp_all_accuracy = (jnp.argmax(
          nsp_logits, -1) == next_sentence_label).astype(nsp_logits.dtype)
      nsp_accuracy = jnp.sum(
          jnp.multiply(nsp_all_accuracy,
                       next_sentence_mask.astype(nsp_logits.dtype)))
      nsp_count = jnp.sum(next_sentence_mask)

    # Outputs
    scalars = {
        'score/eval':
            (mask_accuracy + subregion_accuracy - date_l1_loss * 0.01),
        'loss/eval': mask_loss + date_loss + subregion_loss,
        'loss/date': date_loss,
        'loss/date_l1': date_l1_loss,
        'loss/subregion': subregion_loss,
        'loss/mask': mask_loss,
        'loss/nsp': nsp_loss,
        'count/date': date_count,
        'count/nsp': nsp_count,
        'count/mask': mask_count,
        'accuracy/subregion': subregion_accuracy,
        'accuracy/mask': mask_accuracy,
        'accuracy/nsp': nsp_accuracy,
    }

    outputs = {
        'outputs/id': phi_id,
        'outputs/date_pred': date_pred.astype('float16'),
        'outputs/date_min': date_min,
        'outputs/date_max': date_max,
        'outputs/date_dist': date_dist.astype('float16'),
        'outputs/date_available': date_available,
        'outputs/subregion_logits': subregion_logits.astype('float16'),
        'outputs/subregion': subregion,
    }

    return scalars, outputs, model_log

  def _eval_epoch(self, rng):
    """Evaluates an epoch."""
    summary = {}
    outputs = {}
    total_num_sequences = 0

    # Prepare directories for storing model log
    checkpoint_dir = jl_utils.get_checkpoint_dir(FLAGS.config,
                                                 jax.process_index())
    model_log_path = os.path.join(checkpoint_dir, 'model_log')
    if self.config.evaluation.store_model_log:
      if os.path.isdir(model_log_path):
        map(os.remove, glob.glob(model_log_path + '/*'))
      else:
        os.makedirs(model_log_path)

    # Checkpoints broadcast for each local device
    params = jl_utils.get_first(self._params)

    # Model log buffer initialisation
    model_log_buffer = []

    def _flush_model_log_buffer(model_log_buffer):
      """Writes model log to bz2 pickle files."""
      while model_log_buffer:
        model_log_batch_path, model_log_pkl_bz2 = model_log_buffer.pop(0)
        with open(model_log_batch_path, 'wb') as f:
          f.write(model_log_pkl_bz2)

    # Converting to numpy here allows us to reset the generator
    for batch in self._eval_input():
      # Make sure that the input has batch_dim=1
      assert batch['text_char'].shape[0] == 1

      summary_batch, outputs_batch, model_log_batch = self._eval_batch(
          params, batch, rng)

      # Append batch values to dictionary
      for k, v in summary_batch.items():
        summary[k] = summary.get(k, 0) + v
      for k, v in outputs_batch.items():
        outputs.setdefault(k, []).append(v)

      total_num_sequences += self.config.evaluation.batch_size

      # Store model log per batch
      if self.config.evaluation.store_model_log:
        # Append to buffer
        model_log_batch_path = os.path.join(
            model_log_path,
            str(outputs_batch['outputs/id'][0]) + '.pkl.bz2')
        model_log_pkl = pickle.dumps(model_log_batch, protocol=2)
        model_log_pkl_bz2 = bz2.compress(model_log_pkl)
        model_log_buffer += [(model_log_batch_path, model_log_pkl_bz2)]

        # Flush model log buffer
        if (len(model_log_buffer) %
            self.config.evaluation.store_model_log_steps == 0):
          _flush_model_log_buffer(model_log_buffer)

    # Flush remaining model log buffer
    if self.config.evaluation.store_model_log:
      _flush_model_log_buffer(model_log_buffer)

    # Normalise and concatenate
    summary['loss/date'] /= summary['count/date']
    summary['loss/date_l1'] /= summary['count/date']

    summary['loss/mask'] /= summary['count/mask']
    summary['accuracy/mask'] /= summary['count/mask']

    summary['loss/nsp'] /= summary['count/nsp']
    summary['accuracy/nsp'] /= summary['count/nsp']

    summary['loss/subregion'] /= total_num_sequences
    summary['accuracy/subregion'] /= total_num_sequences

    summary['score/eval'] = (
        summary['accuracy/mask'] + summary['accuracy/subregion'] -
        summary['loss/date_l1'] * 0.01)
    summary['loss/eval'] = (
        summary['loss/mask'] + summary['loss/date'] + summary['loss/subregion'])

    for k, v in outputs.items():
      outputs[k] = np.concatenate(v, axis=0)

    return summary, outputs


if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  app.run(functools.partial(platform.main, Experiment))
