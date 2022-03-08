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
"""Module for performing inference using Jax, including decoding.

The module is separated into two main entrypoints: attribute() and restore().

Both take a function called `forward`, a Jax function mapping from model inputs
(excluding parameters) to the model output tuple. Generated using
e.g. `functools.partial(exp.forward.apply, exp._params)`.
"""

import json
import math
import re
from typing import List, NamedTuple, Tuple

import ithaca.util.eval as eval_util
import ithaca.util.text as util_text

import jax
import numpy as np


class LocationPrediction(NamedTuple):
  """One location prediction and its associated probability."""

  location_id: int
  score: float

  def build_json(self):
    return {
        'location_id': self.location_id,
        'score': self.score,
    }


class AttributionResults(NamedTuple):
  """Immediate model output attribution predictions and related information."""

  input_text: str

  # List of pairs of location ID and probability
  locations: List[LocationPrediction]

  # Probabilities over year range [-800, -790, -780, ..., 790, 800]
  year_scores: List[float]  # length 160

  # Per-character saliency maps:
  date_saliency: List[float]
  location_saliency: List[float]  # originally called subregion

  def build_json(self):
    return {
        'input_text': self.input_text,
        'locations': [l.build_json() for l in self.locations],
        'year_scores': self.year_scores,
        'date_saliency': self.date_saliency,
        'location_saliency': self.location_saliency
    }

  def json(self, **kwargs):
    return json.dumps(self.build_json(), **kwargs)


class Restoration(NamedTuple):
  """One restored candidate string from the beam search."""
  text: str
  score: float

  def build_json(self):
    return {'text': self.text, 'score': self.score}


class RestorationCharSaliency(NamedTuple):
  """Saliency entry for one predicted character of a prediction."""
  text: str
  restored_idx: int  # which predicted character the saliency map corresponds to
  saliency: List[float]

  def build_json(self):
    return {
        'text': self.text,
        'restored_idx': self.restored_idx,
        'saliency': self.saliency
    }


class RestorationResults(NamedTuple):
  """Contains all text-related restoration predictions."""

  input_text: str
  top_prediction: str
  restored: List[int]  # char indices that were missing (-)

  # List of top N results from beam search:
  predictions: List[Restoration]

  # Saliency maps for each successive character of the best (greedy) prediction
  prediction_saliency: List[RestorationCharSaliency]

  def build_json(self):
    return {
        'input_text':
            self.input_text,
        'top_prediction':
            self.top_prediction,
        'restored':
            self.restored,
        'predictions': [r.build_json() for r in self.predictions],
        'prediction_saliency': [
            m.build_json() for m in self.prediction_saliency
        ],
    }

  def json(self, **kwargs):
    return json.dumps(self.build_json(), **kwargs)


# These constants are fixed for all recent versions of the model.
MIN_TEXT_LEN = 50
TEXT_LEN = 768  # fixed model sequence length
DATE_MIN = -800
DATE_MAX = 800
DATE_INTERVAL = 10
RESTORATION_BEAM_WIDTH = 20
RESTORATION_TEMPERATURE = 1.
SEED = 1
ALPHABET_MISSING_RESTORE = '?'  # missing characters to restore


def _prepare_text(
    text, alphabet
) -> Tuple[str, str, str, np.ndarray, np.ndarray, List[int], np.ndarray,
           List[int]]:
  """Adds start of sequence symbol, and padding.

  Also strips accents if present, trims whitespace, and generates arrays ready
  for input into the model.

  Args:
    text: Raw text input string, no padding or start of sequence symbol.
    alphabet: GreekAlphabet object containing index/character mappings.

  Returns:
    Tuple of cleaned text (str), padded text (str), char indices (array of batch
    size 1), word indices (array of batch size 1), text length (list of size 1)
  """
  text = re.sub(r'\s+', ' ', text.strip())
  text = util_text.strip_accents(text)

  if len(text) < MIN_TEXT_LEN:
    raise ValueError('Input text too short.')

  if len(text) >= TEXT_LEN - 1:
    raise ValueError('Input text too long.')

  text_sos = alphabet.sos + text
  text_len = [len(text_sos)]  # includes SOS, but not padding

  text_padded = text_sos + alphabet.pad * max(0, TEXT_LEN - len(text_sos))

  restore_mask_idx = [
      i for i, c in enumerate(text_padded) if c == ALPHABET_MISSING_RESTORE
  ]
  text_padded = text_padded.replace(ALPHABET_MISSING_RESTORE, alphabet.missing)

  text_char = util_text.text_to_idx(text_padded, alphabet).reshape(1, -1)
  text_word = util_text.text_to_word_idx(text_padded, alphabet).reshape(1, -1)
  padding = np.where(text_char > 0, 1, 0)

  return (text, text_sos, text_padded, text_char, text_word, text_len, padding,
          restore_mask_idx)


def attribute(text, forward, params, alphabet, vocab_char_size, vocab_word_size,
              region_map) -> AttributionResults:
  """Computes predicted date and geographical region."""

  (text, _, _, text_char, text_word, text_len, padding,
   _) = _prepare_text(text, alphabet)

  rng = jax.random.PRNGKey(SEED)
  date_logits, subregion_logits, _, _ = forward(
      text_char=text_char,
      text_word=text_word,
      rngs={'dropout': rng},
      is_training=False)

  # Generate subregion predictions:
  subregion_logits = np.array(subregion_logits)
  subregion_pred_probs = eval_util.softmax(subregion_logits[0]).tolist()
  location_predictions = [
      LocationPrediction(location_id=id, score=prob)
      for prob, id in zip(subregion_pred_probs, region_map['sub']['ids'])
  ]
  location_predictions.sort(key=lambda loc: loc.score, reverse=True)

  # Generate date predictions:
  date_pred_probs = eval_util.softmax(date_logits[0])

  # Gradients for saliency maps
  date_saliency, subregion_saliency = eval_util.compute_attribution_saliency_maps(
      text_char, text_word, text_len, padding, forward, params, rng, alphabet,
      vocab_char_size, vocab_word_size)

  # Skip start of sequence symbol (first char) for text and saliency maps:
  return AttributionResults(
      input_text=text,
      locations=location_predictions,
      year_scores=date_pred_probs.tolist(),
      date_saliency=date_saliency.tolist()[1:],
      location_saliency=subregion_saliency.tolist()[1:])


def restore(text, forward, params, alphabet, vocab_char_size,
            vocab_word_size) -> RestorationResults:
  """Performs search to compute text restoration. Slower, runs synchronously."""

  if ALPHABET_MISSING_RESTORE not in text:
    raise ValueError('At least one character must be missing.')

  text, _, text_padded, _, _, text_len, _, restore_mask_idx = _prepare_text(
      text, alphabet)

  beam_result = eval_util.beam_search_batch_2d(
      forward,
      alphabet,
      text_padded,
      restore_mask_idx,
      beam_width=RESTORATION_BEAM_WIDTH,
      temperature=RESTORATION_TEMPERATURE,
      rng=jax.random.PRNGKey(SEED))

  # For visualization purposes, we strip out the SOS and padding, and adjust
  # restored_indices accordingly
  predictions = [
      Restoration(
          text=beam_entry.text_pred[1:].rstrip(alphabet.pad),
          score=math.exp(beam_entry.pred_logprob)) for beam_entry in beam_result
  ]
  restored_indices = [i - 1 for i in restore_mask_idx]

  # Sequence of saliency maps for a greedy prediction:
  saliency_steps = eval_util.sequential_restoration_saliency(
      text_padded, text_len, forward, params, alphabet, restore_mask_idx,
      vocab_char_size, vocab_word_size)

  return RestorationResults(
      input_text=text,
      top_prediction=predictions[0].text,
      restored=restored_indices,
      predictions=predictions,
      prediction_saliency=[
          RestorationCharSaliency(step.text, int(step.pred_char_pos),
                                  step.saliency_map.tolist())
          for step in saliency_steps
      ])
