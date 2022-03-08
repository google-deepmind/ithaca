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
"""Date processing functions."""
import numpy as np


def date_num_bins(date_min, date_max, date_interval, unknown_bin=True):
  num_bins = (date_max - date_min - 1) // date_interval
  if unknown_bin:
    num_bins += 1  # +1 for unk
  return num_bins


def date_to_bin(date_cur, date_min, date_max, date_interval, date_bins):
  if date_cur >= date_min and date_cur < date_max:
    date_bin = np.digitize(
        date_cur,
        list(range(date_min + date_interval, date_max, date_interval)))
  else:
    date_bin = date_bins - 1
  return date_bin


def bin_to_date(date_cur_bin, date_min, date_interval):
  return date_min + date_cur_bin * date_interval + date_interval // 2


def date_range_to_dist(date_min_cur,
                       date_max_cur,
                       date_min,
                       date_max,
                       date_interval,
                       date_bins,
                       return_logits=True):
  """Converts a date range to a uniform distribution."""
  dist = np.zeros(date_bins)

  if (date_min_cur and date_max_cur and date_min_cur >= date_min and
      date_max_cur < date_max and date_min_cur <= date_max_cur):
    date_min_cur_bin = date_to_bin(date_min_cur, date_min, date_max,
                                   date_interval, date_bins)
    date_max_cur_bin = date_to_bin(date_max_cur, date_min, date_max,
                                   date_interval, date_bins)
  else:
    date_min_cur_bin = date_bins - 1
    date_max_cur_bin = date_bins - 1

  date_bins_cur = date_max_cur_bin - date_min_cur_bin + 1
  dist[date_min_cur_bin:date_max_cur_bin + 1] = 1. / date_bins_cur

  if return_logits:
    eps = 1e-6
    dist = np.clip(dist, eps, 1. - eps)
    dist = np.log(dist)

  return dist
