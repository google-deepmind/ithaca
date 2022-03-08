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
"""Subregion mapping used to train the model.

The subregion IDs originate from the I.PHI generator and may be subject to
change in future versions of the PHI dataset.
"""


def load_region_maps(region_file):
  """Extracts creates a map from PHI region id to a continuous region id."""
  region_ids = []  # Used mainly for eval
  region_ids_inv = {}  # Used in data loader
  region_names_inv = {}  # Used in eval
  for l in region_file.read().strip().split('\n'):
    tok_name_id, _ = l.strip().split(';')  # second field is frequency, unused
    region_name, region_id = tok_name_id.split('_')
    region_name = region_name.strip()
    region_id = int(region_id)
    # Ignore unknown regions:
    if ((region_name == 'Unknown Provenances' and region_id == 884) or
        (region_name == 'unspecified subregion' and region_id == 885) or
        (region_name == 'unspecified subregion' and region_id == 1439)):
      continue
    region_ids.append(region_id)
    region_ids_inv[region_id] = len(region_ids_inv)
    region_names_inv[len(region_names_inv)] = region_name

  return {
      'ids': region_ids,
      'ids_inv': region_ids_inv,
      'names_inv': region_names_inv
  }
