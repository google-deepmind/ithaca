# Copyright 2021 the Ithaca Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dataloader functions."""

import json
import random
import re

from absl import logging
from ithaca.util.dates import date_range_to_dist
from ithaca.util.text import random_mask_span
from ithaca.util.text import random_sentence_swap
from ithaca.util.text import random_word_delete
from ithaca.util.text import random_word_swap
from ithaca.util.text import text_to_idx
from ithaca.util.text import text_to_word_idx
import numpy as np
import tensorflow.compat.v1 as tf


def generate_sample(config, alphabet, region_map, sample, mode='train'):
  """Generates a new TF dataset sample."""

  # Get text
  text = sample['text']

  # Next sentence prediction
  sentences = text.split('.')
  # Strip spaces
  sentences = list(map(str.strip, sentences))
  # Filter blank sentences
  sentences = list(filter(None, sentences))
  # Generate indexes
  sentence_idx = np.arange(len(sentences), dtype=np.int32)

  # Random sentence shuffling
  if (mode == 'train' and config.random_sentence_swap > 0):
    # Shuffle indexes
    sentence_idx = random_sentence_swap(sentence_idx,
                                        config.random_sentence_swap)
    # Reshuffle sentences
    sentences = np.array(sentences)[sentence_idx].tolist()

  # Random word swap
  if mode == 'train' and config.random_word_swap > 0:
    sentences = [
        random_word_swap(s, config.random_word_swap) for s in sentences
    ]

  # Random word delete
  if mode == 'train' and config.random_word_delete > 0:
    sentences = [
        random_word_delete(s, config.random_word_delete) for s in sentences
    ]

  # Join text
  text = '. '.join(sentences) + '.'

  # Generate mask and labels
  next_sentence_dots = np.array(
      [pos for pos, char in enumerate(text[:-1]) if char == '.'],
      dtype=np.int32)
  next_sentence_mask = np.zeros(len(text), dtype=bool)
  next_sentence_label = np.zeros(len(text), dtype=np.int32)
  if sentence_idx.size > 1:
    next_sentence_mask[next_sentence_dots] = True
    next_sentence_label[next_sentence_dots] = (
        sentence_idx[:-1] == (sentence_idx[1:] - 1))

  # Computer start for prepending start of sentence character
  start_sample_idx = int(config.prepend_sos)

  if (mode in ['train', 'valid'] and config.context_char_random and
      len(text) >= config.context_char_min):
    # During training pick random context length
    context_char_len = np.random.randint(
        config.context_char_min,
        min(len(text), config.context_char_max - start_sample_idx) + 1)

    start_idx = 0
    if context_char_len < len(text):
      start_idx = np.random.randint(0, len(text) - context_char_len + 1)
    text = text[start_idx:start_idx + context_char_len - start_sample_idx]
    next_sentence_mask = next_sentence_mask[start_idx:start_idx +
                                            context_char_len - start_sample_idx]
    next_sentence_label = next_sentence_label[start_idx:start_idx +
                                              context_char_len -
                                              start_sample_idx]
  elif (config.context_char_max and len(text) >
        (config.context_char_max - start_sample_idx)):
    # Clip text by maximum length
    start_idx = np.random.randint(
        0,
        len(text) - (config.context_char_max - start_sample_idx) + 1)
    text = text[start_idx:start_idx + config.context_char_max -
                start_sample_idx]
    next_sentence_mask = next_sentence_mask[start_idx:start_idx +
                                            config.context_char_max -
                                            start_sample_idx]
    next_sentence_label = next_sentence_label[start_idx:start_idx +
                                              config.context_char_max -
                                              start_sample_idx]

  # Prepend start of sentence character
  if config.prepend_sos:
    text = alphabet.sos + text
    next_sentence_mask = [False] + next_sentence_mask
    next_sentence_label = [0] + next_sentence_label

  # Unmasked text
  text_unmasked_idx = text_to_idx(text, alphabet)
  text_unmasked_word_idx = text_to_word_idx(text, alphabet)

  # Mask text
  text_mask = np.zeros(len(text), dtype=bool)
  if mode in ['train', 'valid']:
    text_list = list(text)

    # Non missing idx (avoid removing start of sentence character)
    non_missing_idx = []
    for i in range(start_sample_idx, len(text_list)):
      if text_list[i] not in [alphabet.missing] + alphabet.punctuation:
        non_missing_idx.append(i)

    # Skip sample if there are no usable characters
    if not non_missing_idx:
      return

    char_mask_idx = []
    if config.char_mask_rate_max > 0.:
      # Compute rate
      char_mask_rate = np.random.uniform(config.char_mask_rate_min,
                                         config.char_mask_rate_max)

      # Fix masking in valid mode for comparing experiments
      span_mask_geometric_p = config.span_mask_geometric_p
      mask_num_total = int(char_mask_rate * len(non_missing_idx))
      mask_num_span = int(mask_num_total * config.span_mask_ratio)
      if mode == 'valid' and config.span_mask_eval_len > 0:
        span_mask_geometric_p = None
        mask_num_total = min(config.span_mask_eval_len, len(non_missing_idx))
        mask_num_span = mask_num_total
      mask_num_char = mask_num_total - mask_num_span

      # Mask random indices
      if mask_num_char > 0:
        char_mask_idx = np.random.choice(
            non_missing_idx, mask_num_char, replace=False).tolist()

      # Mask random spans
      if mask_num_span > 0:
        count_span = 0
        span_mask_idx = []
        while (len(span_mask_idx) < mask_num_span and count_span < 10000):
          span_mask_idx.extend(
              random_mask_span(
                  text,
                  geometric_p=span_mask_geometric_p,
                  limit_chars=mask_num_span - len(span_mask_idx)))
          count_span += 1
        char_mask_idx.extend(span_mask_idx)

    # Mask text
    for idx in set(char_mask_idx):
      text_mask[idx] = True
      text_list[idx] = alphabet.missing
    text = ''.join(text_list)

  # Text missing mask
  text_missing_mask = np.array(list(text)) == alphabet.missing

  # Convert to indices
  text_idx = text_to_idx(text, alphabet)
  text_idx_len = len(text_idx)
  text_word_idx = text_to_word_idx(text, alphabet)
  text_word_idx_len = len(text_word_idx)
  assert text_idx_len == text_word_idx_len

  # PHI id
  phi_id = int(sample['id'])

  # Map region ids to local ids
  region_main_id = region_map['main']['ids_inv'][int(sample['region_main_id'])]
  region_sub_id = region_map['sub']['ids_inv'][int(sample['region_sub_id'])]

  # Dates
  if (sample['date_min'] and sample['date_max'] and
      int(sample['date_min']) <= int(sample['date_max']) and
      int(sample['date_min']) >= config.date_min and
      int(sample['date_max']) < config.date_max):
    date_available = True
    date_min = float(sample['date_min'])
    date_max = float(sample['date_max'])
    date_dist = date_range_to_dist(date_min, date_max, config.date_min,
                                   config.date_max, config.date_interval,
                                   config.date_bins)
  else:
    date_available = False
    date_min = 0.
    date_max = 0.
    date_dist = date_range_to_dist(None, None, config.date_min, config.date_max,
                                   config.date_interval, config.date_bins)

  return {
      'id': phi_id,  # 'text_str': text,
      'text_char': text_idx,
      'text_mask': text_mask,
      'text_missing_mask': text_missing_mask,
      'text_word': text_word_idx,
      'text_len': text_idx_len,
      'text_unmasked': text_unmasked_idx,
      'text_unmasked_word': text_unmasked_word_idx,
      'next_sentence_mask': next_sentence_mask,
      'next_sentence_label': next_sentence_label,
      'region_main_id': region_main_id,
      'region_sub_id': region_sub_id,
      'date_available': date_available,
      'date_min': date_min,
      'date_max': date_max,
      'date_dist': date_dist,
  }


def loader_tf(batch_size,
              config,
              region_map,
              alphabet=None,
              dataset_file=None,
              mode='train'):
  """TF dataloader."""
  # Load dataset
  dataset_tmp = {int(d['id']): d for d in json.load(dataset_file)}
  logging.info('Loaded dataset inscriptions: %d.', len(dataset_tmp))

  # Check if white_list enabled
  if hasattr(config, 'white_list') and config.white_list:
    dataset = []
    for d in dataset_tmp.values():
      if int(d['id']) in config.white_list:
        dataset.append(d)
    del dataset_tmp
  else:
    # Find duplicate inscriptions
    rev_dataset = {}
    black_list = set()
    if hasattr(config, 'black_list') and config.black_list:
      logging.info('Ignore list inscriptions: %d.', len(config.black_list))
      black_list.update(config.black_list)

    for key in sorted(dataset_tmp.keys()):
      value = dataset_tmp[key]
      rev_dataset.setdefault(value['text'], set()).add(key)
      if len(rev_dataset[value['text']]) > 1:
        black_list.add(int(value['id']))
    del rev_dataset
    logging.info('Inscriptions filtered: %d.', len(black_list))

    # Create deduplicated dataset
    dataset = []
    for d in dataset_tmp.values():
      if int(d['id']) not in black_list:
        dataset.append(d)
    del dataset_tmp
    del black_list

  logging.info('Final dataset inscriptions: %d.', len(dataset))

  # Breaks dataset correlated order.
  random.shuffle(dataset)

  # Sample generator function
  def generate_samples():

    dataset_idxs = list(range(len(dataset)))
    random.shuffle(dataset_idxs)
    for dataset_i in dataset_idxs:
      sample = dataset[dataset_i]

      # Skip if region does not exist in map
      if (int(sample['region_main_id']) not in region_map['main']['ids_inv'] or
          int(sample['region_sub_id']) not in region_map['sub']['ids_inv']):
        continue

      # Replace guess signs with missing chars
      if hasattr(config, 'char_use_guess') and not config.char_use_guess:
        sample['text'] = re.sub(r'\[(.*?)\]', lambda m: '-' * len(m.group(1)),
                                sample['text'])
      sample['text'] = sample['text'].replace(alphabet.sog,
                                              '').replace(alphabet.eog, '')

      # Filter by text length
      if len(sample['text'].replace(alphabet.missing,
                                    '')) < config.context_char_min:
        continue

      # Last digit 3 -> test, 4 -> valid, the rest are the training set
      sample_id = int(sample['id'])
      if ((sample_id % 10 == 3 and mode == 'test') or
          (sample_id % 10 == 4 and mode == 'valid') or
          (sample_id % 10 != 3 and sample_id % 10 != 4 and mode == 'train') or
          (hasattr(config, 'white_list') and config.white_list)):
        s = generate_sample(config, alphabet, region_map, sample, mode=mode)
        if s:
          yield s

  # Create dataset from generator.
  with tf.device('/cpu:0'):
    ds = tf.data.Dataset.from_generator(
        generate_samples,
        output_signature={
            'id':
                tf.TensorSpec(shape=(), dtype=tf.int32),
            'text_char':
                tf.TensorSpec(shape=(None), dtype=tf.int32),
            'text_mask':
                tf.TensorSpec(shape=(None), dtype=tf.bool),
            'text_missing_mask':
                tf.TensorSpec(shape=(None), dtype=tf.bool),
            'text_word':
                tf.TensorSpec(shape=(None), dtype=tf.int32),
            'text_unmasked':
                tf.TensorSpec(shape=(None), dtype=tf.int32),
            'text_unmasked_word':
                tf.TensorSpec(shape=(None), dtype=tf.int32),
            'next_sentence_mask':
                tf.TensorSpec(shape=(None), dtype=tf.bool),
            'next_sentence_label':
                tf.TensorSpec(shape=(None), dtype=tf.int32),
            'text_len':
                tf.TensorSpec(shape=(), dtype=tf.int32),
            'region_main_id':
                tf.TensorSpec(shape=(), dtype=tf.int32),
            'region_sub_id':
                tf.TensorSpec(shape=(), dtype=tf.int32),
            'date_available':
                tf.TensorSpec(shape=(), dtype=tf.bool),
            'date_min':
                tf.TensorSpec(shape=(), dtype=tf.float32),
            'date_max':
                tf.TensorSpec(shape=(), dtype=tf.float32),
            'date_dist':
                tf.TensorSpec(shape=(config.date_bins), dtype=tf.float32),
        })

  # Shuffle and repeat.
  if mode == 'train':
    if config.repeat_train == -1:
      ds = ds.repeat()
    elif config.repeat_train >= 1:
      ds = ds.repeat(config.repeat_train)
  else:
    if config.repeat_eval == -1:
      ds = ds.repeat()
    elif config.repeat_eval >= 1:
      ds = ds.repeat(config.repeat_eval)

  # Batch and pad.
  max_len = config.context_char_max
  ds = ds.padded_batch(
      batch_size,
      padded_shapes={
          'id': [],
          'text_char': [max_len],
          'text_mask': [max_len],
          'text_missing_mask': [max_len],
          'text_word': [max_len],
          'text_unmasked': [max_len],
          'text_unmasked_word': [max_len],
          'next_sentence_mask': [max_len],
          'next_sentence_label': [max_len],
          'text_len': [],
          'region_main_id': [],
          'region_sub_id': [],
          'date_available': [],
          'date_min': [],
          'date_max': [],
          'date_dist': [config.date_bins]
      },
      padding_values={
          'id': 0,
          'text_char': alphabet.pad_idx,
          'text_mask': False,
          'text_missing_mask': True,
          'text_word': alphabet.pad_idx,
          'text_unmasked': alphabet.pad_idx,
          'text_unmasked_word': alphabet.pad_idx,
          'next_sentence_mask': False,
          'next_sentence_label': 0,
          'text_len': 0,
          'region_main_id': 0,
          'region_sub_id': 0,
          'date_available': False,
          'date_min': 0.,
          'date_max': 0.,
          'date_dist': 0.
      })

  return ds
