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
"""Text processing functions."""

import random
import re
import unicodedata

import numpy as np


def idx_to_text(idxs, alphabet, strip_sos=True, strip_pad=True):
  """Converts a list of indices to a string."""
  idxs = np.array(idxs)
  out = ''
  for i in range(idxs.size):
    idx = idxs[i]
    if strip_pad and idx == alphabet.pad_idx:
      break
    elif strip_sos and idx == alphabet.sos_idx:
      pass
    else:
      out += alphabet.idx2char[idx]
  return out


def idx_to_text_batch(idxs, alphabet, lengths=None):
  """Converts batched lists of indices to strings."""
  b = []
  for i in range(idxs.shape[0]):
    idxs_i = idxs[i]
    if lengths:
      idxs_i = idxs_i[:lengths[i]]
    b.append(idx_to_text(idxs_i, alphabet))
  return b


def random_mask_span(t, geometric_p=0.2, limit_chars=None):
  """Masks a span of sequential words."""

  # Obtain span indexes (indlusive)
  span_idx = [(ele.start(), ele.end()) for ele in re.finditer(r'[\w\s]+', t)]
  if not span_idx:
    return []

  # Select a span to mask
  span_start, span_end = random.choice(span_idx)

  # Sample a random span length using a geomteric distribution
  if geometric_p and limit_chars:
    span_len = np.clip(
        np.random.geometric(geometric_p),
        1, min(limit_chars, span_end - span_start))
  elif geometric_p:
    span_len = np.clip(
        np.random.geometric(geometric_p),
        1, span_end - span_start)
  elif limit_chars:
    span_len = min(limit_chars, span_end - span_start)
  else:
    raise ValueError('geometric_p or limit_chars should be set.')

  # Pick a random start index
  span_start = np.random.randint(span_start, span_end - span_len + 1)
  assert span_start + span_len <= span_end

  # Clip to limit chars
  if limit_chars is not None and span_len >= limit_chars:
    span_len = limit_chars

  # Create mask indices
  mask_idx = list(range(span_start, span_start + span_len))

  return mask_idx


def random_sentence_swap(sentences, p):
  """Swaps sentences with probability p."""

  def swap_sentence(s):
    idx_1 = random.randint(0, len(s) - 1)
    idx_2 = idx_1
    counter = 0

    while idx_2 == idx_1:
      idx_2 = random.randint(0, len(s) - 1)
      counter += 1
      if counter > 3:
        return s

    s[idx_1], s[idx_2] = s[idx_2], s[idx_1]
    return s

  new_sentences = sentences.copy()
  n = int(p * len(sentences))
  for _ in range(n):
    new_sentences = swap_sentence(new_sentences)

  return new_sentences


def random_word_delete(sentence, p):
  """Deletes a word from a sentence with probability p."""

  words = sentence.split(' ')

  # Return if one word.
  if len(words) == 1:
    return words[0]

  # Randomly delete words.
  new_words = []
  for word in words:
    if random.uniform(0, 1) > p:
      new_words.append(word)

  # If all words are removed return one.
  if not new_words:
    rand_int = random.randint(0, len(words) - 1)
    return words[rand_int]

  sentence = ' '.join(new_words)

  return sentence


def random_word_swap(sentence, p):
  """Swaps words from a sentence with probability p."""

  def swap_word(new_words):
    idx_1 = random.randint(0, len(new_words) - 1)
    idx_2 = idx_1
    counter = 0

    while idx_2 == idx_1:
      idx_2 = random.randint(0, len(new_words) - 1)
      counter += 1

      if counter > 3:
        return new_words

    new_words[idx_1], new_words[idx_2] = new_words[idx_2], new_words[idx_1]
    return new_words

  words = sentence.split(' ')

  new_words = words.copy()
  n = int(p * len(words))
  for _ in range(n):
    new_words = swap_word(new_words)

  sentence = ' '.join(new_words)

  return sentence


def strip_accents(s):
  return ''.join(
      c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')


def text_to_idx(t, alphabet):
  """Converts a string to character indices."""
  return np.array([alphabet.char2idx[c] for c in t], dtype=np.int32)


def text_to_word_idx(t, alphabet):
  """Converts a string to word indices."""
  out = np.full(len(t), alphabet.word2idx[alphabet.unk], dtype=np.int32)
  for m in re.finditer(r'\w+', t):
    if m.group() in alphabet.word2idx:
      out[m.start():m.end()] = alphabet.word2idx[m.group()]
  return out

