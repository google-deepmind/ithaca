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
"""Alphabet classes."""

import re

import numpy as np


class Alphabet:
  """Generic alphabet class."""

  def __init__(self,
               alphabet,
               numerals='0',
               punctuation='.',
               space=' ',
               missing='-',
               pad='#',
               unk='^',
               sos='<',
               sog='[',
               eog=']',
               wordlist_file=None,
               wordlist_size=100000):
    self.alphabet = list(alphabet)  # alph
    self.numerals = list(numerals)  # num
    self.punctuation = list(punctuation)  # punt
    self.space = space  # spacing
    self.missing = missing  # missing char
    self.pad = pad  # padding (spaces to right of string)
    self.unk = unk  # unknown char
    self.sos = sos  # start of sentence
    self.sog = sog  # start of guess
    self.eog = eog  # end of guess

    # Define wordlist mapping
    idx2word = [self.pad, self.sos, self.unk]
    if wordlist_file:
      idx2word += [
          w_c.split(';')[0]
          for w_c in wordlist_file.read().strip().split('\n')[:wordlist_size]
      ]
    self.idx2word = np.array(idx2word)
    self.word2idx = {self.idx2word[i]: i for i in range(len(self.idx2word))}

    # Define vocab mapping
    self.idx2char = np.array(
        [self.pad, self.sos, self.unk, self.space, self.missing] +
        self.alphabet + self.numerals + self.punctuation)
    self.char2idx = {self.idx2char[i]: i for i in range(len(self.idx2char))}

    # Define special character indices
    self.pad_idx = self.char2idx[pad]
    self.sos_idx = self.char2idx[sos]
    self.unk_idx = self.char2idx[unk]
    self.alphabet_start_idx = self.char2idx[self.alphabet[0]]
    self.alphabet_end_idx = self.char2idx[self.numerals[-1]]

  def filter(self, t):
    return t

  def size_char(self):
    return len(self.idx2char)

  def size_word(self):
    return len(self.idx2word)


class GreekAlphabet(Alphabet):
  """Greek alphabet class."""

  def __init__(self, wordlist_file=None, wordlist_size=100000):
    greek_alphabet = 'αβγδεζηθικλμνξοπρςστυφχψωϙϛ'

    super().__init__(
        alphabet=greek_alphabet,
        wordlist_file=wordlist_file,
        wordlist_size=wordlist_size)
    self.tonos_to_oxia = {
        # tonos  : #oxia
        u'\u0386': u'\u1FBB',  # capital letter alpha
        u'\u0388': u'\u1FC9',  # capital letter epsilon
        u'\u0389': u'\u1FCB',  # capital letter eta
        u'\u038C': u'\u1FF9',  # capital letter omicron
        u'\u038A': u'\u1FDB',  # capital letter iota
        u'\u038E': u'\u1FF9',  # capital letter upsilon
        u'\u038F': u'\u1FFB',  # capital letter omega
        u'\u03AC': u'\u1F71',  # small letter alpha
        u'\u03AD': u'\u1F73',  # small letter epsilon
        u'\u03AE': u'\u1F75',  # small letter eta
        u'\u0390': u'\u1FD3',  # small letter iota with dialytika and tonos/oxia
        u'\u03AF': u'\u1F77',  # small letter iota
        u'\u03CC': u'\u1F79',  # small letter omicron
        u'\u03B0': u'\u1FE3',
        # small letter upsilon with dialytika and tonos/oxia
        u'\u03CD': u'\u1F7B',  # small letter upsilon
        u'\u03CE': u'\u1F7D'  # small letter omega
    }
    self.oxia_to_tonos = {v: k for k, v in self.tonos_to_oxia.items()}

  def filter(self, t):  # override previous filter function
    # lowercase
    t = t.lower()

    # replace dot below
    t = t.replace(u'\u0323', '')

    # replace perispomeni
    t = t.replace(u'\u0342', '')
    t = t.replace(u'\u02C9', '')

    # replace ending sigma
    t = re.sub(r'([\w\[\]])σ(?![\[\]])(\b)', r'\1ς\2', t)

    # replace oxia with tonos
    for oxia, tonos in self.oxia_to_tonos.items():
      t = t.replace(oxia, tonos)

    # replace h
    h_patterns = {
        # input: #target
        'ε': 'ἑ',
        'ὲ': 'ἓ',
        'έ': 'ἕ',
        'α': 'ἁ',
        'ὰ': 'ἃ',
        'ά': 'ἅ',
        'ᾶ': 'ἇ',
        'ι': 'ἱ',
        'ὶ': 'ἳ',
        'ί': 'ἵ',
        'ῖ': 'ἷ',
        'ο': 'ὁ',
        'ό': 'ὅ',
        'ὸ': 'ὃ',
        'υ': 'ὑ',
        'ὺ': 'ὓ',
        'ύ': 'ὕ',
        'ῦ': 'ὗ',
        'ὴ': 'ἣ',
        'η': 'ἡ',
        'ή': 'ἥ',
        'ῆ': 'ἧ',
        'ὼ': 'ὣ',
        'ώ': 'ὥ',
        'ω': 'ὡ',
        'ῶ': 'ὧ'
    }

    # iterate by keys
    for h_in, h_tar in h_patterns.items():
      # look up and replace h[ and h]
      t = re.sub(r'ℎ(\[?){}'.format(h_in), r'\1{}'.format(h_tar), t)
      t = re.sub(r'ℎ(\]?){}'.format(h_in), r'{}\1'.format(h_tar), t)

    # any h left is an ἡ
    t = re.sub(r'(\[?)ℎ(\]?)', r'\1ἡ\2', t)

    return t
