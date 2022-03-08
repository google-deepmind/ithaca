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
"""Eval utils."""

from typing import List, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from .text import idx_to_text
from .text import text_to_idx
from .text import text_to_word_idx
import tqdm


def date_loss_l1(pred, target_min, target_max):
  """L1 loss function for dates."""
  loss = 0.
  loss += np.abs(pred - target_min) * np.less(pred, target_min).astype(
      pred.dtype)
  loss += np.abs(pred - target_max) * np.greater(pred, target_max).astype(
      pred.dtype)
  return loss


def grad_to_saliency_char(gradient_char, text_char_onehot, text_len, alphabet):
  """Generates saliency map."""
  saliency_char = np.linalg.norm(gradient_char, axis=2)[0, :text_len[0]]

  text_char = np.array(text_char_onehot).argmax(axis=-1)
  idx_mask = np.logical_or(
      text_char[0, :text_len[0]] > alphabet.alphabet_end_idx,
      text_char[0, :text_len[0]] < alphabet.alphabet_start_idx)
  idx_unmask = np.logical_not(idx_mask)

  saliency_char_tmp = saliency_char.copy()
  saliency_char_tmp[idx_mask] = 0.
  if idx_unmask.any():
    saliency_char_tmp[idx_unmask] = (saliency_char[idx_unmask] -
                                     saliency_char[idx_unmask].min()) / (
                                         saliency_char[idx_unmask].max() -
                                         saliency_char[idx_unmask].min() + 1e-8)
  return saliency_char_tmp


def grad_to_saliency_word(gradient_word, text_word_onehot, text_len, alphabet):
  """Generates saliency map."""
  saliency_word = np.linalg.norm(gradient_word, axis=2)[0, :text_len[0]]
  text_word = np.array(text_word_onehot).argmax(axis=-1)

  saliency_word = saliency_word.copy()
  start_idx = None
  for i in range(text_len[0]):
    if text_word[0, i] == alphabet.unk_idx:
      if start_idx is not None:
        saliency_word[start_idx:i] = np.sum(saliency_word[start_idx:i])
      start_idx = None
    elif start_idx is None:
      start_idx = i

  idx_mask = text_word[0, :text_len[0]] == alphabet.unk_idx
  idx_unmask = np.logical_not(idx_mask)
  saliency_word_tmp = saliency_word.copy()
  saliency_word_tmp[idx_mask] = 0.
  if idx_unmask.any():
    saliency_word_tmp[idx_unmask] = (
        saliency_word[idx_unmask] - saliency_word[idx_unmask].min())
    saliency_word_tmp[idx_unmask] = saliency_word_tmp[idx_unmask] / (
        saliency_word[idx_unmask].max() - saliency_word[idx_unmask].min() +
        1e-8)
  return saliency_word_tmp


def softmax(x, axis=-1):
  """Compute softmax values for each sets of scores in x."""
  unnormalized = np.exp(x - x.max(axis, keepdims=True))
  return unnormalized / unnormalized.sum(axis, keepdims=True)


def log_softmax(x, axis=-1):
  """Log-Softmax function."""
  shifted = x - x.max(axis, keepdims=True)
  return shifted - np.log(np.sum(np.exp(shifted), axis, keepdims=True))


def nucleus_sample_inner(logits, top_p=0.95, temp=1.0):
  """Samples from the most likely tokens whose probability sums to top_p."""
  sorted_logits = np.sort(logits)
  sorted_probs = softmax(sorted_logits)
  threshold_idx = np.argmax(np.cumsum(sorted_probs, -1) >= 1 - top_p)
  threshold_largest_logits = sorted_logits[..., [threshold_idx]]
  assert threshold_largest_logits.shape == logits.shape[:-1] + (1,)
  mask = logits >= threshold_largest_logits
  logits += (1 - mask) * -1e12  # Set unused logits to -inf.
  logits /= np.maximum(temp, 1e-12)
  return logits


class BeamEntry(NamedTuple):
  text_pred: str
  mask_idx: int
  pred_len: int
  pred_logprob: float


def beam_search_batch_2d(forward,
                         alphabet,
                         text_pred,
                         mask_idx,
                         rng=None,
                         beam_width=20,
                         temperature=1.,
                         nucleus=False,
                         nucleus_top_p=0.8,
                         display_progress=False) -> List[BeamEntry]:
  """Non-sequential beam search."""

  beam = [BeamEntry(text_pred, mask_idx, 0, 0.)]
  beam_top = {}

  text_len = len(text_pred.rstrip(alphabet.pad))

  # Initialise tqdm bar
  if display_progress:
    pbar = tqdm.tqdm(total=len(mask_idx))

  while beam:
    beam_tmp = []
    beam_batch = []

    text_chars = []
    text_words = []

    for text_pred, mask_idx, pred_len, pred_logprob in beam:

      mask_idx = mask_idx.copy()  # pytype: disable=attribute-error  # strict_namedtuple_checks
      text_char = text_to_idx(text_pred, alphabet).reshape(1, -1)
      text_word = text_to_word_idx(text_pred, alphabet).reshape(1, -1)
      text_chars.append(text_char)
      text_words.append(text_word)
      beam_batch.append(BeamEntry(text_pred, mask_idx, pred_len, pred_logprob))
    text_chars = np.vstack(text_chars)
    text_words = np.vstack(text_words)

    _, _, mask_logits, _ = forward(
        text_char=text_chars,
        text_word=text_words,
        text_char_onehot=None,
        text_word_onehot=None,
        rngs={'dropout': rng},
        is_training=False)
    mask_logits = mask_logits / temperature
    mask_logits = np.array(mask_logits)

    for batch_i in range(mask_logits.shape[0]):
      text_pred, mask_idx, pred_len, pred_logprob = beam_batch[batch_i]
      mask_logprob = log_softmax(mask_logits)[batch_i, :text_len]
      mask_pred = softmax(mask_logits)[batch_i, :text_len]
      mask_pred_argmax = np.dstack(
          np.unravel_index(np.argsort(-mask_pred.ravel()), mask_pred.shape))[0]

      # Keep only predictions for mask
      for i in range(mask_pred_argmax.shape[0]):
        if (mask_pred_argmax[i][0] in mask_idx and  # pytype: disable=unsupported-operands  # strict_namedtuple_checks
            (mask_pred_argmax[i][1] == alphabet.char2idx[alphabet.space] or
             (mask_pred_argmax[i][1] >= alphabet.alphabet_start_idx and
              mask_pred_argmax[i][1] <=
              alphabet.char2idx[alphabet.punctuation[-1]]))):
          text_char_i = text_chars.copy()
          text_char_i[batch_i, mask_pred_argmax[i][0]] = mask_pred_argmax[i][1]
          text_pred_i = idx_to_text(
              text_char_i[batch_i], alphabet, strip_sos=False, strip_pad=False)

          mask_idx_i = mask_idx.copy()  # pytype: disable=attribute-error  # strict_namedtuple_checks
          mask_idx_i.remove(mask_pred_argmax[i][0])

          if nucleus:
            mask_logits_i = mask_logits[batch_i, mask_pred_argmax[i][0]]
            mask_logits_i = nucleus_sample_inner(mask_logits_i, nucleus_top_p)
            mask_logprob_i = log_softmax(mask_logits_i)

            # Skip expanding the beam if logprob too small
            if mask_logits_i[mask_pred_argmax[i][1]] < -1e12:
              continue

            pred_logprob_i = pred_logprob + mask_logprob_i[mask_pred_argmax[i]
                                                           [1]]
          else:
            pred_logprob_i = pred_logprob + mask_logprob[mask_pred_argmax[i][0],
                                                         mask_pred_argmax[i][1]]

          if not mask_idx_i:
            if (text_pred_i
                not in beam_top) or (text_pred_i in beam_top and
                                     beam_top[text_pred_i][3] > pred_logprob_i):
              beam_top[text_pred_i] = BeamEntry(text_pred_i, mask_idx_i,
                                                pred_len + 1, pred_logprob_i)
          else:
            beam_tmp.append(
                BeamEntry(text_pred_i, mask_idx_i, pred_len + 1,
                          pred_logprob_i))

    # order all candidates by score
    beam_tmp_kv = {}
    for text_pred, mask_idx, pred_len, pred_logprob in beam_tmp:
      if (text_pred not in beam_tmp_kv) or (
          text_pred in beam_tmp_kv and
          beam_tmp_kv[text_pred].pred_logprob > pred_logprob):
        beam_tmp_kv[text_pred] = BeamEntry(text_pred, mask_idx, pred_len,
                                           pred_logprob)
    beam_tmp = sorted(
        beam_tmp_kv.values(),
        key=lambda entry: entry.pred_logprob,
        reverse=True)

    # select k best
    beam = beam_tmp[:beam_width]

    # update progress bar
    if display_progress:
      pbar.update(1)

  # order all candidates by score
  return sorted(
      beam_top.values(), key=lambda entry: entry.pred_logprob,
      reverse=True)[:beam_width]


def beam_search_batch_1d(forward,
                         alphabet,
                         text_pred,
                         mask_idx,
                         rng=None,
                         beam_width=20,
                         temperature=1.,
                         nucleus=False,
                         nucleus_top_p=0.8,
                         display_progress=False) -> List[BeamEntry]:
  """Sequential beam search."""

  beam = [BeamEntry(text_pred, mask_idx, 0, 0.)]
  beam_top = {}

  # Initialise tqdm bar
  if display_progress:
    pbar = tqdm.tqdm(total=len(mask_idx))

  while beam:
    beam_tmp = []
    beam_batch = []

    text_chars = []
    text_words = []

    for text_pred, mask_idx, pred_len, pred_logprob in beam:

      mask_idx = mask_idx.copy()  # pytype: disable=attribute-error  # strict_namedtuple_checks
      text_char = text_to_idx(text_pred, alphabet).reshape(1, -1)
      text_word = text_to_word_idx(text_pred, alphabet).reshape(1, -1)
      text_chars.append(text_char)
      text_words.append(text_word)
      beam_batch.append(BeamEntry(text_pred, mask_idx, pred_len, pred_logprob))
    text_chars = np.vstack(text_chars)
    text_words = np.vstack(text_words)

    _, _, mask_logits, _ = forward(
        text_char=text_chars,
        text_word=text_words,
        text_char_onehot=None,
        text_word_onehot=None,
        rngs={'dropout': rng},
        is_training=False)
    mask_logits = mask_logits / temperature
    mask_logits = np.array(mask_logits)

    for batch_i in range(mask_logits.shape[0]):
      text_pred, mask_idx, pred_len, pred_logprob = beam_batch[batch_i]

      mask_logits_i = mask_logits[batch_i, mask_idx[0]]  # pytype: disable=unsupported-operands  # strict_namedtuple_checks
      if nucleus:
        mask_logits_i = nucleus_sample_inner(mask_logits_i, nucleus_top_p)

      mask_logprob = log_softmax(mask_logits_i)

      # Keep only predictions for mask
      alphabet_chars = [alphabet.char2idx[alphabet.space]]
      alphabet_chars += list(
          range(alphabet.alphabet_start_idx,
                alphabet.char2idx[alphabet.punctuation[-1]]))
      for char_i in alphabet_chars:
        # Skip expanding the beam if logprob too small
        if nucleus and mask_logits_i[char_i] < -1e12:
          continue

        text_char_i = text_chars.copy()
        text_char_i[batch_i, mask_idx[0]] = char_i  # pytype: disable=unsupported-operands  # strict_namedtuple_checks

        text_pred_i = idx_to_text(
            text_char_i[batch_i], alphabet, strip_sos=False, strip_pad=False)

        mask_idx_i = mask_idx.copy()  # pytype: disable=attribute-error  # strict_namedtuple_checks
        mask_idx_i.pop(0)
        pred_logprob_i = pred_logprob + mask_logprob[char_i]

        if not mask_idx_i:
          if (text_pred_i
              not in beam_top) or (text_pred_i in beam_top and
                                   beam_top[text_pred_i][3] > pred_logprob_i):
            beam_top[text_pred_i] = BeamEntry(text_pred_i, mask_idx_i,
                                              pred_len + 1, pred_logprob_i)
        else:
          beam_tmp.append(
              BeamEntry(text_pred_i, mask_idx_i, pred_len + 1, pred_logprob_i))

    # order all candidates by score
    beam_tmp_kv = {}
    for text_pred, mask_idx, pred_len, pred_logprob in beam_tmp:
      if (text_pred
          not in beam_tmp_kv) or (text_pred in beam_tmp_kv and
                                  beam_tmp_kv[text_pred][3] > pred_logprob):
        beam_tmp_kv[text_pred] = BeamEntry(text_pred, mask_idx, pred_len,
                                           pred_logprob)
    beam_tmp = sorted(
        beam_tmp_kv.values(),
        key=lambda entry: entry.pred_logprob,
        reverse=True)

    # select k best
    beam = beam_tmp[:beam_width]

    # update progress bar
    if display_progress:
      pbar.update(1)

  # order all candidates by score
  return sorted(
      beam_top.values(), key=lambda entry: entry.pred_logprob,
      reverse=True)[:beam_width]


def saliency_loss_subregion(forward,
                            text_char_emb,
                            text_word_emb,
                            padding,
                            rng,
                            subregion=None):
  """Saliency map for subregion."""

  _, subregion_logits, _, _ = forward(
      text_char_emb=text_char_emb,
      text_word_emb=text_word_emb,
      padding=padding,
      rngs={'dropout': rng},
      is_training=False)
  if subregion is None:
    subregion = subregion_logits.argmax(axis=-1)[0]
  return subregion_logits[0, subregion]


def saliency_loss_date(forward, text_char_emb, text_word_emb, padding, rng):
  """saliency_loss_date."""

  date_pred, _, _, _ = forward(
      text_char_emb=text_char_emb,
      text_word_emb=text_word_emb,
      padding=padding,
      rngs={'dropout': rng},
      is_training=False)

  date_pred_argmax = date_pred.argmax(axis=-1)
  return date_pred[0, date_pred_argmax[0]]


def predicted_dates(date_pred_probs, date_min, date_max, date_interval):
  """Returns mode and mean prediction."""
  date_years = np.arange(date_min + date_interval / 2,
                         date_max + date_interval / 2, date_interval)

  # Compute mode:
  date_pred_argmax = (
      date_pred_probs.argmax() * date_interval + date_min + date_interval // 2)

  # Compute mean:
  date_pred_avg = np.dot(date_pred_probs, date_years)

  return date_pred_argmax, date_pred_avg


def compute_attribution_saliency_maps(text_char,
                                      text_word,
                                      text_len,
                                      padding,
                                      forward,
                                      params,
                                      rng,
                                      alphabet,
                                      vocab_char_size,
                                      vocab_word_size,
                                      subregion_loss_kwargs=None):
  """Compute saliency maps for subregions and dates."""

  if subregion_loss_kwargs is None:
    subregion_loss_kwargs = {}

  # Get saliency gradients
  dtype = params['params']['char_embeddings']['embedding'].dtype
  text_char_onehot = jax.nn.one_hot(text_char, vocab_char_size).astype(dtype)
  text_word_onehot = jax.nn.one_hot(text_word, vocab_word_size).astype(dtype)
  text_char_emb = jnp.matmul(text_char_onehot,
                             params['params']['char_embeddings']['embedding'])
  text_word_emb = jnp.matmul(text_word_onehot,
                             params['params']['word_embeddings']['embedding'])
  gradient_subregion_char, gradient_subregion_word = jax.grad(
      saliency_loss_subregion, (1, 2))(
          forward,
          text_char_emb,
          text_word_emb,
          padding,
          rng=rng,
          **subregion_loss_kwargs)
  gradient_date_char, gradient_date_word = jax.grad(saliency_loss_date, (1, 2))(
      forward, text_char_emb, text_word_emb, padding=padding, rng=rng)

  # Generate saliency maps for subregions
  input_grad_subregion_char = np.multiply(gradient_subregion_char,
                                          text_char_emb)  # grad x input
  input_grad_subregion_word = np.multiply(gradient_subregion_word,
                                          text_word_emb)
  grad_char = grad_to_saliency_char(
      input_grad_subregion_char,
      text_char_onehot,
      text_len=text_len,
      alphabet=alphabet)
  grad_word = grad_to_saliency_word(
      input_grad_subregion_word,
      text_word_onehot,
      text_len=text_len,
      alphabet=alphabet)
  subregion_saliency = np.clip(grad_char + grad_word, 0, 1)

  # Generate saliency maps for dates
  input_grad_date_char = np.multiply(gradient_date_char,
                                     text_char_emb)  # grad x input
  input_grad_date_word = np.multiply(gradient_date_word, text_word_emb)
  grad_char = grad_to_saliency_char(
      input_grad_date_char,
      text_char_onehot,
      text_len=text_len,
      alphabet=alphabet)
  grad_word = grad_to_saliency_word(
      input_grad_date_word,
      text_word_onehot,
      text_len=text_len,
      alphabet=alphabet)
  date_saliency = np.clip(grad_char + grad_word, 0, 1)

  return date_saliency, subregion_saliency


def saliency_loss_mask(forward, text_char_emb, text_word_emb, padding, rng,
                       char_pos, char_idx):
  """Saliency map for mask."""

  _, _, mask_logits, _ = forward(
      text_char_emb=text_char_emb,
      text_word_emb=text_word_emb,
      text_char_onehot=None,
      text_word_onehot=None,
      padding=padding,
      rngs={'dropout': rng},
      is_training=False)
  return mask_logits[0, char_pos, char_idx]


class SequentialRestorationSaliencyResult(NamedTuple):
  text: str  # predicted text string so far
  pred_char_pos: int  # newly restored character's position
  saliency_map: np.ndarray  # saliency map for the newly added character


def sequential_restoration_saliency(text_str, text_len, forward, params,
                                    alphabet, mask_idx, vocab_char_size,
                                    vocab_word_size):
  """Greedily, non-sequentially restores, producing per-step saliency maps."""
  text_len = text_len[0] if not isinstance(text_len, int) else text_len
  rng = jax.random.PRNGKey(0)  # dummy, no randomness in model
  mask_idx = set(mask_idx)
  while mask_idx:
    text_char = text_to_idx(text_str, alphabet).reshape(1, -1)
    padding = jnp.where(text_char > 0, 1, 0)
    text_word = text_to_word_idx(text_str, alphabet).reshape(1, -1)

    _, _, mask_logits, _ = forward(
        text_char=text_char,
        text_word=text_word,
        text_char_onehot=None,
        text_word_onehot=None,
        rngs={'dropout': rng},
        is_training=False)
    mask_pred = jax.nn.softmax(mask_logits)[0, :text_len]
    mask_pred_argmax = np.dstack(
        np.unravel_index(np.argsort(-mask_pred.ravel()), mask_pred.shape))[0]

    # Greedily, non-sequentially take the next highest probability prediction
    # out of the characters that are to be restored
    for i in range(mask_pred_argmax.shape[0]):
      pred_char_pos, pred_char_idx = mask_pred_argmax[i]
      if pred_char_pos in mask_idx:
        break

    # Update sequence
    text_char[0, pred_char_pos] = pred_char_idx
    text_str = idx_to_text(
        text_char[0], alphabet, strip_sos=False, strip_pad=False)
    mask_idx.remove(pred_char_pos)

    # Gradients for saliency map
    text_char_onehot = jax.nn.one_hot(text_char,
                                      vocab_char_size).astype(jnp.float32)
    text_word_onehot = jax.nn.one_hot(text_word,
                                      vocab_word_size).astype(jnp.float32)

    text_char_emb = jnp.matmul(text_char_onehot,
                               params['params']['char_embeddings']['embedding'])
    text_word_emb = jnp.matmul(text_word_onehot,
                               params['params']['word_embeddings']['embedding'])

    gradient_mask_char, gradient_mask_word = jax.grad(
        saliency_loss_mask, (1, 2))(
            forward,
            text_char_emb,
            text_word_emb,
            padding,
            rng=rng,
            char_pos=pred_char_pos,
            char_idx=pred_char_idx)

    # Use gradient x input for visualizing saliency
    input_grad_mask_char = np.multiply(gradient_mask_char, text_char_emb)
    input_grad_mask_word = np.multiply(gradient_mask_word, text_word_emb)

    # Return visualization-ready saliency maps
    saliency_map = grad_to_saliency_char(
        np.clip(input_grad_mask_char + input_grad_mask_word, 0, 1),
        text_char_onehot, [text_len], alphabet)  # normalize, etc.
    result_text = idx_to_text(text_char[0], alphabet, strip_sos=False)  # no pad

    yield SequentialRestorationSaliencyResult(
        text=result_text[1:],
        pred_char_pos=pred_char_pos - 1,
        saliency_map=saliency_map[1:])
