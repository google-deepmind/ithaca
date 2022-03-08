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
"""Config for a Ithaca experiment."""

from jaxline import base_config
from ml_collections import config_dict


def get_config():
  """Return config object for training."""

  config = base_config.get_base_config()

  # Experiment config.
  # Modify this to adapt to your custom distributed learning setup
  local_batch_size = 1
  num_devices = 1
  config.train_batch_size = local_batch_size * num_devices

  # Experiment config.
  config.macros = config_dict.ConfigDict(
      dict(
          wordlist_size=35884,  # Keeping words with freq >10
          context_char_max=768,
          date_max=800,
          date_min=-800,
          date_interval=10,
          date_bins=160,
      ))
  cm = config.macros  # Alias.

  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              random_seed=4,
              random_mode_train=config.get_ref('random_mode_train'),
              random_mode_eval=config.get_ref('random_mode_eval'),
              optimizer=dict(
                  name='lamb',
                  kwargs=dict(
                      learning_rate=3e-4,
                      weight_decay=0.,
                      b2=0.999,
                  ),
                  # Set up the learning rate schedule.
                  # factors='constant * linear_warmup * rsqrt_decay',
                  warmup=4000,
                  clip_adaptive=False,
                  clip_level=0.,
              ),
              training=dict(
                  batch_size=config.get_oneway_ref('train_batch_size')),
              alphabet=dict(
                  wordlist_path='data/iphi-wordlist.txt',
                  wordlist_size=cm.get_ref('wordlist_size'),
              ),
              dataset=dict(
                  dataset_path='data/iphi.json',
                  region_main_path='data/iphi-region-main.txt',
                  region_sub_path='data/iphi-region-sub.txt',
                  context_char_min=50,
                  context_char_max=cm.get_ref('context_char_max'),
                  context_char_random=True,
                  char_use_guess=True,
                  char_mask_rate_min=0.,
                  char_mask_rate_max=0.5,
                  span_mask_eval_len=10,
                  span_mask_ratio=0.15,
                  span_mask_geometric_p=0.1,
                  random_sentence_swap=0.25,
                  random_word_delete=0.2,
                  random_word_swap=0.,
                  date_min=cm.get_ref('date_min'),
                  date_max=cm.get_ref('date_max'),
                  date_interval=cm.get_ref('date_interval'),
                  date_bins=cm.get_ref('date_bins'),
                  prepend_sos=1,
                  repeat_train=-1,
                  repeat_eval=10,
                  black_list=[
                      # 2334, 10, 293931, 14, 293752, 15, 293753, 16, 11,
                      # 294468, 229647, 12, 291324, 291317, 17, 232697, 293754,
                      # 1682, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 291118,
                      # 291320, 291319, 292366, 34, 291960, 35, 32, 346490, 27,
                      # 292187, 291318, 19, 18, 37, 291321, 292189, 293756, 42,
                      # 46, 232710, 39, 40, 41, 291322, 293757, 293327, 28,
                      # 292194, 293326, 21, 293755, 291319, 291117, 38, 291959,
                      # 31, 232705
                  ],
                  white_list=[]),
              model=dict(
                  word_char_emb_dim=256,
                  emb_dim=512,
                  mlp_dim=2048,
                  num_layers=8,
                  num_heads=4,
                  vocab_char_size=34,
                  vocab_word_size=cm.get_ref('wordlist_size') + 4,
                  output_subregions=85,
                  output_date=cm.get_ref('date_bins'),
                  output_date_dist=True,
                  region_date_pooling='first',
                  use_output_mlp=True,
                  max_len=cm.get_ref('context_char_max'),
                  dropout_rate=0.1,
                  attention_dropout_rate=0.1,
                  use_bfloat16=False,
                  model_type='bigbird',
                  feature_combine_type='concat',
                  posemb_combine_type='concat',
              ),
              loss=dict(
                  date=dict(
                      enabled=True,
                      type='dist',
                      weight_dist=1.25,
                      weight_l1=0.,
                      label_smoothing=0.,
                      step_start=0,
                      step_end=0,
                  ),
                  region=dict(
                      enabled=True,
                      weight=2.,
                      label_smoothing=0.1,
                      step_start=0,
                      step_end=0,
                  ),
                  mask=dict(
                      enabled=True,
                      weight=3.,
                      label_smoothing=0.05,
                      step_start=0,
                      step_end=0,
                  ),
                  nsp=dict(
                      enabled=True,
                      weight=0.01,
                      step_start=0,
                      step_end=0,
                  )),
              evaluation=dict(
                  use_jit=True,
                  batch_size=1,
                  mode='valid',
                  store_model_log=False,
                  store_model_log_steps=100,
              ),
          ),))

  # Training loop config.
  config.training_steps = 1_000_000
  config.log_train_data_interval = 10
  config.save_checkpoint_interval = 300
  config.best_model_eval_metric = 'score/eval'
  config.checkpoint_dir = '/tmp/ithaca_checkpoints'
  config.train_checkpoint_all_hosts = False

  # Prevents accidentally setting keys that aren't recognized (e.g. in tests).
  config.lock()

  return config
