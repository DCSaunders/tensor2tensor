# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""BLEU metric util used during eval for MT."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import re
import sys
import time
import unicodedata

# Dependency imports

import numpy as np
import six
# pylint: disable=redefined-builtin
from six.moves import xrange
from six.moves import zip
# pylint: enable=redefined-builtin

import pyter
import tensorflow as tf
from tensor2tensor.utils import decoding

from tensor2tensor.utils import bleu_hook

BLEU_MAX_ORDER=4

def empty_max_order_per_set(set_count):
  return [get_empty_max_order() for _ in range(set_count)]

def get_empty_max_order():
  return BLEU_MAX_ORDER * [0]


def get_bleu_brevity_penalty(hyp_len, ref_len):
  if ref_len:
    ratio = max(hyp_len, 1) / ref_len
    bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
  else:
    bp = 1.0
  return bp

def seq_bleus(samples, targets):
  scales = []
  for s, t in zip(samples, targets):
    ref_ngrams, ref_len = get_trimmed_ngrams_and_len(t)
    sample_ngrams, sample_len = get_trimmed_ngrams_and_len(s)
    bleu = get_sentence_bleu(ref_ngrams, ref_len, sample_ngrams, sample_len)
    scales.append([bleu])
  return np.asarray(scales, dtype=np.float32)


def get_ordered_batch_gleus(samples, targets, inputs, num_sets, do_ordering, hyps, matches, lens,
                            sample_set_orders):
  all_ref_len = 0
  hyp_matches = hyp_hyps = hyp_lens = None
  for idx, (s, t) in enumerate(zip(samples, targets)):
    set_idx = idx % num_sets
    target_idx = int(idx / num_sets)
    ref_ngrams, ref_len = get_trimmed_ngrams_and_len(t)
    if set_idx == 0:
      all_ref_len += ref_len
      hyp_matches = empty_max_order_per_set(num_sets)
      hyp_hyps = empty_max_order_per_set(num_sets)
      hyp_lens = []
    hyp_ngrams, hyp_len = get_trimmed_ngrams_and_len(s)
    source_ngrams, _ = get_trimmed_ngrams_and_len(inputs[target_idx])
    not_src_ngrams = get_not_src_ngrams(ref_ngrams, source_ngrams)
    if do_ordering:
      not_src_ngram_matches(hyp_ngrams, not_src_ngrams, hyp_matches[set_idx], hyp_hyps[set_idx])
      hyp_lens.append(hyp_len)
      if set_idx == num_sets - 1: # last sample for a given target
        target_samples = samples[idx - set_idx : idx + 1]
        reorder_sample_sets(target_samples,
                                 hyp_matches,
                                 hyp_hyps,
                                 hyp_lens,
                                 matches,
                                 hyps,
                                 lens,
                                 sample_set_orders,
                                 target_idx,
                                 ref_len)
    else:
      not_src_ngram_matches(hyp_ngrams, not_src_ngrams, matches[set_idx], hyps[set_idx])
      lens[set_idx] += hyp_len
  return all_ref_len

def get_trimmed_ngrams_and_len(seq):
  trimmed_seq = decoding._save_until_eos(seq, is_image=False)
  ngrams = bleu_hook._get_ngrams(trimmed_seq, BLEU_MAX_ORDER)
  return ngrams, len(trimmed_seq)


def get_ordered_batch_bleus(samples, targets, num_sets, do_ordering, hyps, matches, lens,
                            sample_set_orders):
  all_ref_len = 0
  hyp_matches = hyp_hyps = hyp_lens = None
  for idx, (s, t) in enumerate(zip(samples, targets)):
    set_idx = idx % num_sets
    target_idx = int(idx / num_sets)
    ref_ngrams, ref_len = get_trimmed_ngrams_and_len(t)
    if set_idx == 0:
      all_ref_len += ref_len
      hyp_matches = empty_max_order_per_set(num_sets)
      hyp_hyps = empty_max_order_per_set(num_sets)
      hyp_lens = []
    hyp_ngrams, hyp_len = get_trimmed_ngrams_and_len(s)
    if do_ordering:
      ngram_matches(hyp_ngrams, ref_ngrams, hyp_matches[set_idx], hyp_hyps[set_idx])
      hyp_lens.append(hyp_len)
      if set_idx == num_sets - 1: # last sample for a given target
        target_samples = samples[idx - set_idx : idx + 1]
        reorder_sample_sets(target_samples,
                                 hyp_matches,
                                 hyp_hyps,
                                 hyp_lens,
                                 matches,
                                 hyps,
                                 lens,
                                 sample_set_orders,
                                 target_idx,
                                 ref_len)
    else:
      ngram_matches(hyp_ngrams, ref_ngrams, matches[set_idx], hyps[set_idx])
      lens[set_idx] += hyp_len
  return all_ref_len


def reorder_sample_sets(samples, hyp_matches, hyp_hyps, hyp_lens,  
                        matches, hyps, lens,
                        sample_set_orders, target_idx, ref_len):
  total_bleus = [get_bleu_from_matches(h, m, l, ref_len)
                 for h, m, l in zip(hyp_hyps, hyp_matches, hyp_lens)]
  #if hparams.mrt_zero_bleu_high:
  #  total_bleus = [1-b for b in total_bleus]
  set_idx = 0
  for ordered_idx, _ in sorted(enumerate(total_bleus), key=lambda x: x[1]):
    sample_set_orders[target_idx].append(ordered_idx)
    lens[set_idx] += hyp_lens[ordered_idx]
    for order in range(BLEU_MAX_ORDER):
      matches[set_idx][order] += hyp_matches[ordered_idx][order]
      hyps[set_idx][order] += hyp_hyps[ordered_idx][order]
    set_idx += 1

def reorder_sample_sets_ter(samples, hyp_edits, edits,
                        sample_set_orders, target_idx, ref_len):
  set_idx = 0
  for ordered_idx, _ in sorted(enumerate(hyp_edits), key=lambda x: x[1]):
    # if sample is identical to last, use id of last set to sample_set_orders
    sample_set_orders[target_idx].append(ordered_idx)
    edits[set_idx] += hyp_edits[ordered_idx]
    set_idx += 1


def batch_bleus(samples, targets, num_sets, do_ordering, inputs=None):
  """
  If we have N samples per source sentence, take the N-wise sample sets and find the average set bleu metric
  We need two quantities for our metric:
  - The average set bleu across all N possible sets
  - The set bleu for each possible set
  !! this makes a lot more sense if the samples are ordered somehow !!
  """
  set_hyps = empty_max_order_per_set(num_sets)
  set_matches = empty_max_order_per_set(num_sets)
  num_per_set = int(len(targets) / num_sets)
  set_lens = {idx: 0 for idx in range(num_sets)}
  set_sample_orders = [[] for _ in range(num_per_set)] # in order, for each set, id of samples corresponding to the best set, 2nd best etc
  if inputs is not None:
    ref_len = get_ordered_batch_gleus(samples, targets, inputs, num_sets, do_ordering, set_hyps, set_matches, set_lens, set_sample_orders)
  else:
    ref_len = get_ordered_batch_bleus(samples, targets, num_sets, do_ordering, set_hyps, set_matches, set_lens, set_sample_orders)
  set_bleus = []
  for set_idx in range(num_sets):
    set_bleu = get_bleu_from_matches(set_hyps[set_idx], set_matches[set_idx], set_lens[set_idx], ref_len)
    set_bleus.append(set_bleu)
  if do_ordering:
    scales = []
    for sample_order in set_sample_orders:
      reordered_scales = np.zeros_like(sample_order, dtype=np.float32)
      for idx, l in enumerate(sample_order):
        reordered_scales[l] = set_bleus[idx]
      scales.append(reordered_scales)
    scales = np.asarray(scales, dtype=np.float32)
  else:
    set_bleus = np.asarray(set_bleus, dtype=np.float32)
    scales = np.tile(set_bleus, num_per_set)
  scales = scales.reshape([scales.size, 1])
  return scales


def ngram_matches(hyp_ngrams, ref_ngrams, matches, hyps):
  for ngram, count in hyp_ngrams.items():
    hyps[len(ngram) - 1] += count
  for ngram, count in ref_ngrams.items():
    if hyp_ngrams[ngram] > 0:
      matches[len(ngram) - 1] += min(count, hyp_ngrams[ngram])

def get_not_src_ngrams(ref_ngrams, source_ngrams):
  ref_not_source = collections.Counter()
  for ngram, count in ref_ngrams.items():
    if count - source_ngrams[ngram] > 0:
      ref_not_source[ngram] = count - source_ngrams[ngram]
  return ref_not_source

def not_src_ngram_matches(hyp_ngrams, ref_not_src_ngrams, matches, hyps):
  for ngram, count in ref_not_src_ngrams.items():
    if hyp_ngrams[ngram] > 0:
      matches[len(ngram) - 1] += min(count, hyp_ngrams[ngram])
    hyps[len(ngram) - 1] += count

def get_batch_ters(samples, targets, num_sets, do_ordering):
  scales = []
  set_edits = [0 for _ in range(num_sets)]
  num_per_set = int(len(targets) / num_sets)
  set_sample_orders = [[] for _ in range(num_per_set)]
  ref_len = get_ordered_batch_ters(samples, targets, num_sets, do_ordering, set_edits, set_sample_orders)
  set_ters = []
  for set_idx in range(num_sets):
    set_ter = set_edits[set_idx] / ref_len
    set_ters.append(set_ter)
  if do_ordering:
    scales = []
    for sample_order in set_sample_orders:
      reordered_scales = np.zeros_like(sample_order, dtype=np.float32)
      for idx, l in enumerate(sample_order):
        reordered_scales[l] = set_ters[idx]
      scales.append(reordered_scales)
    scales = np.asarray(scales, dtype=np.float32)
  else:
    scales = np.tile(set_ters, num_per_set)
  scales = scales.reshape([scales.size, 1])
  return scales


def get_ordered_batch_ters(samples, targets, num_sets, do_ordering, edits, sample_set_orders):
  all_ref_len = 0
  hyp_edits = None
  for idx, (s, t) in enumerate(zip(samples, targets)):
    set_idx = idx % num_sets
    target_idx = int(idx / num_sets)
    ref = decoding._save_until_eos(t, is_image=False)        
    ref_len = len(ref)
    sample = decoding._save_until_eos(s, is_image=False)
    num_edits = pyter.ter(sample, ref) * ref_len
    if set_idx == 0:
      all_ref_len += ref_len
      hyp_edits = [0 for _ in range(num_sets)]
    if do_ordering:
      hyp_edits[set_idx] += num_edits
      if set_idx == num_sets - 1: # last sample for a given target
        target_samples = samples[idx - set_idx : idx + 1]
        reorder_sample_sets_ter(target_samples,
                                hyp_edits,
                                edits,
                                sample_set_orders,
                                target_idx,
                                ref_len)
    else:
      edits[set_idx] += num_edits
  return all_ref_len

def get_sentence_bleu(ref_ngrams, ref_len, hyp_ngrams, hyp_len):
  matches_by_order = get_empty_max_order()
  hyps_by_order = get_empty_max_order()
  ngram_matches(hyp_ngrams, ref_ngrams, matches_by_order, hyps_by_order)
  bleu = get_bleu_from_matches(hyps_by_order, matches_by_order, hyp_len, ref_len, smoothing=True)
  return bleu


def get_ters(samples, targets, num_sets=None):
  scales = []
  for s, t in zip(samples, targets):
    ref = decoding._save_until_eos(t, is_image=False)        
    sample = decoding._save_until_eos(s, is_image=False)
    scales.append(pyter.ter(sample, ref))
  return np.asarray(scales, dtype=np.float32)


def get_bleu_from_matches(hyps_by_order, matches_by_order, hyp_len, ref_len, smoothing=False, brevity_penalty=True):
  smooth = 1.0
  precisions = get_empty_max_order()
  if not sum(hyps_by_order):
    return 0.0
  for i in xrange(BLEU_MAX_ORDER):
    if hyps_by_order[i]:
      if matches_by_order[i]:
        precisions[i] = matches_by_order[i] / hyps_by_order[i]
      elif smoothing:
        smooth *= 2
        precisions[i] = 1.0 / (smooth * hyps_by_order[i])
  bleu = math.exp(sum(math.log(p) for p in precisions if p) / BLEU_MAX_ORDER)
  if brevity_penalty:
    bleu *= get_bleu_brevity_penalty(hyp_len, ref_len)
  return bleu
