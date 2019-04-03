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

"""transformer (attention).

encoder: [Self-Attention, Feed-forward] x n
decoder: [Self-Attention, Source-Target-Attention, Feed-forward] x n
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import decoding
from tensor2tensor.utils import bleu_hook
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf
import numpy as np
import math
import pyter
import copy
import collections
from tensorflow.python.util import nest


@registry.register_model
class Transformer(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  def __init__(self, *args, **kwargs):
    super(Transformer, self).__init__(*args, **kwargs)
    self.attention_weights = dict()  # For vizualizing attention heads.

  def encode(self, inputs, target_space, hparams, features=None):
    """Encode transformer inputs.

    Args:
      inputs: Transformer inputs [batch_size, input_length, input_height,
        hidden_dim] which will be flattened along the two spatial dimensions.
      target_space: scalar, target space ID.
      hparams: hyperparmeters for model.
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encodre-decoder attention. [batch_size, input_length]
    """
    inputs = common_layers.flatten4d3d(inputs)
    
    encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
        transformer_prepare_encoder(
            inputs, target_space, hparams, features=features))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    encoder_output = transformer_encoder(
        encoder_input, self_attention_bias,
        hparams, nonpadding=features_to_nonpadding(features, "inputs"),
        save_weights_to=self.attention_weights)

    return encoder_output, encoder_decoder_attention_bias

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             nonpadding=None):
    """Decode Transformer outputs from encoder representation.

    Args:
      decoder_input: inputs to bottom of the model.
          [batch_size, decoder_length, hidden_dim]
      encoder_output: Encoder representation.
          [batch_size, input_length, hidden_dim]
      encoder_decoder_attention_bias: Bias and mask weights for
          encoder-decoder attention. [batch_size, input_length]
      decoder_self_attention_bias: Bias and mask weights for decoder
          self-attention. [batch_size, decoder_length]
      hparams: hyperparmeters for model.
      cache: dict, containing tensors which are the results of previous
          attentions, used for fast decoding.
      nonpadding: optional Tensor with shape [batch_size, decoder_length]

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)
    decoder_output = transformer_decoder(
        decoder_input,
        encoder_output,
        decoder_self_attention_bias,
        encoder_decoder_attention_bias,
        hparams,
        cache=cache,
        nonpadding=nonpadding,
        save_weights_to=self.attention_weights)

    if (common_layers.is_on_tpu() and
        hparams.mode == tf.estimator.ModeKeys.TRAIN):
      # TPU does not react kindly to extra dimensions.
      # TODO(noam): remove this once TPU is more forgiving of extra dims.
      return decoder_output
    else:
      # Expand since t2t expects 4d tensors.
      return tf.expand_dims(decoder_output, axis=2)

  def body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
          "tragets": Target decoder outputs.
              [batch_size, decoder_length, hidden_dim]
          "target_space_id"

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    inputs = features.get("inputs")
    encoder_output, encoder_decoder_attention_bias = (None, None)
    if inputs is not None and 'inputs_raw' in features:
      target_space = features["target_space_id"]
      encoder_output, encoder_decoder_attention_bias = self.encode(
          inputs, target_space, hparams, features=features)
    targets = features["targets"]
    targets = common_layers.flatten4d3d(targets)

    decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
        targets, hparams, features=features)
    
    return self.decode(decoder_input, encoder_output,
                       encoder_decoder_attention_bias,
                       decoder_self_attention_bias, hparams,
                       nonpadding=features_to_nonpadding(features, "targets"))

  def _greedy_infer(self, features, decode_length):
    """Fast version of greedy decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    with tf.variable_scope(self.name):
      return self._fast_decode(features, decode_length)

  def _beam_decode(self, features, decode_length, beam_size, top_beams, alpha):
    """Beam search decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
    """
    with tf.variable_scope(self.name):
      return self._fast_decode(
          features, decode_length, beam_size, top_beams, alpha)

  def _fast_decode(self,
                   features,
                   decode_length,
                   beam_size=1,
                   top_beams=1,
                   alpha=1.0):
    """Fast decoding.

    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams
    inputs = features["inputs"]
    target_modality = self._problem_hparams.target_modality
    if target_modality.is_class_modality:
      decode_length = 1
    else:
      decode_length = common_layers.shape_list(inputs)[1] + decode_length

    # TODO(llion): Clean up this reshaping logic.
    inputs = tf.expand_dims(inputs, axis=1)
    if len(inputs.shape) < 5:
      inputs = tf.expand_dims(inputs, axis=4)
    s = common_layers.shape_list(inputs)
    inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
    # _shard_features called to ensure that the variable names match

    inputs = self._shard_features({"inputs": inputs})["inputs"]
    input_modality = self._problem_hparams.input_modality["inputs"]
    with tf.variable_scope(input_modality.name):
      inputs = input_modality.bottom_sharded(inputs, dp)

    with tf.variable_scope("body"):
      encoder_output, encoder_decoder_attention_bias = dp(
          self.encode, inputs, features["target_space_id"], hparams,
          features=features)
    encoder_output = encoder_output[0]
    encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]

    if hparams.pos == "timing":
      timing_signal = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)

    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: inputs ids to the decoder. [batch_size, 1]
        i: scalar, Step number of the decoding loop.

      Returns:
        Processed targets [batch_size, 1, hidden_dim]
      """
      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      with tf.variable_scope(target_modality.name):
        targets = target_modality.targets_bottom_sharded(targets, dp)[0]
      targets = common_layers.flatten4d3d(targets)

      # TODO(llion): Explain! Is this even needed?
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      if hparams.pos == "timing":
        targets += timing_signal[:, i:i + 1]
      return targets

    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
          decode_length)

    def symbols_to_logits_fn(ids, i, cache):
      """Go from ids to logits for next symbol."""
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)
      bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
      with tf.variable_scope("body"):
        body_outputs = dp(
            self.decode, targets, cache["encoder_output"],
            cache["encoder_decoder_attention_bias"], bias, hparams, cache,
            nonpadding=features_to_nonpadding(features, "targets"))

      with tf.variable_scope(target_modality.name):
        logits = target_modality.top_sharded(body_outputs, None, dp)[0]

      return tf.squeeze(logits, axis=[1, 2, 3]), cache

    return fast_decode(
        encoder_output=encoder_output,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        symbols_to_logits_fn=symbols_to_logits_fn,
        hparams=hparams,
        decode_length=decode_length,
        vocab_size=target_modality.top_dimensionality,
        beam_size=beam_size,
        top_beams=top_beams,
        alpha=alpha)


  def _fast_decode_nodp(self,
                        features,
                        decode_length,
                        beam_size=1,
                        top_beams=1,
                        alpha=1.0):
    """Fast decoding.

    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams
    inputs = features["inputs"]
    target_modality = self._problem_hparams.target_modality
    if target_modality.is_class_modality:
      decode_length = 1
    elif self.hparams.mrt_autoregressive_sample or self.hparams.mrt_beam_sample:
      decode_length = decode_length
    else:
      decode_length = common_layers.shape_list(inputs)[1] + decode_length

    # TODO(llion): Clean up this reshaping logic.
    inputs = tf.expand_dims(inputs, axis=1)
    if len(inputs.shape) < 5:
      inputs = tf.expand_dims(inputs, axis=4)
    s = common_layers.shape_list(inputs)
    inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
    # _shard_features called to ensure that the variable names match

    #inputs = self._shard_features({"inputs": inputs})["inputs"]
    input_modality = self._problem_hparams.input_modality["inputs"]
    with tf.variable_scope(input_modality.name):
      #inputs = input_modality.bottom_sharded(inputs, dp)
      inputs = input_modality.bottom(inputs)

    with tf.variable_scope("body"):
      encoder_output, encoder_decoder_attention_bias = self.encode(inputs, features["target_space_id"], hparams,
          features=features)

    if hparams.pos == "timing":
      timing_signal = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)

    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: inputs ids to the decoder. [batch_size, 1]
        i: scalar, Step number of the decoding loop.

      Returns:
        Processed targets [batch_size, 1, hidden_dim]
      """
      # _shard_features called to ensure that the variable names match
      #targets = self._shard_features({"targets": targets})["targets"]
      with tf.variable_scope(target_modality.name):
        #targets = target_modality.targets_bottom_sharded(targets, dp)[0]
        targets = target_modality.targets_bottom(targets)
      targets = common_layers.flatten4d3d(targets)

      # TODO(llion): Explain! Is this even needed?
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      if hparams.pos == "timing":
        targets += timing_signal[:, i:i + 1]
      return targets

    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
          decode_length)

    def symbols_to_logits_fn(ids, i, cache):
      """Go from ids to logits for next symbol."""
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)
      bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      with tf.variable_scope("body"):
        body_outputs = self.decode(targets, cache["encoder_output"],
            cache["encoder_decoder_attention_bias"], bias, hparams, cache,
            nonpadding=features_to_nonpadding(features, "targets"))

      with tf.variable_scope(target_modality.name):
        logits = target_modality.top(body_outputs, None)

      return tf.squeeze(logits, axis=[1, 2, 3]), cache

    return fast_decode(
        encoder_output=encoder_output,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        symbols_to_logits_fn=symbols_to_logits_fn,
        hparams=hparams,
        decode_length=decode_length,
        vocab_size=target_modality.top_dimensionality,
        beam_size=beam_size,
        top_beams=top_beams,
        alpha=alpha)

  def forward_pass_features(self, features):
    transformed_features = self.bottom(features)
    with tf.variable_scope("body", reuse=tf.AUTO_REUSE):
      tf.logging.info("Forward pass through model body")
      body_out = self.body(transformed_features)
    output, losses = self._normalize_body_output(body_out)
    return output, losses

  def tile_with_adjacent_repeats(self, t, tile_num):
    shape = common_layers.shape_list(t)
    tiled = tf.tile(t, [1, tile_num] + [1] * (len(shape) - 2))
    reshaped_tiled = tf.reshape(tiled, [shape[0] * tile_num] + shape[1:])
    return reshaped_tiled

  def mask_samples(self, samples, targets):
    pad_len = len(targets[0].squeeze())
    samples_out = []
    for idx, unstripped_sample in enumerate(samples):
      sample = decoding._save_until_eos(unstripped_sample, is_image=False)
      if sum(sample) == 0 and self.hparams.mrt_beam_sample:
        # sometimes beam search returns 0s when beams do not finish
        # in this case just tile the target
        padded_sample = targets[idx].squeeze()
      else:
        padded_sample = np.zeros(pad_len)
        padded_sample[:sample.size] = sample
        if sample.size < pad_len:
          padded_sample[sample.size] = 1       
      samples_out.append(padded_sample.reshape([pad_len, 1, 1]))
    #tf.logging.info(np.asarray(samples_out, dtype=np.int32).squeeze())
    return np.asarray(samples_out, dtype=np.int32)

  def get_features_samples_and_inputs(self, features, logits):
    if 'sequence_scale' not in features:
      tf.logging.warning('Discriminative training requires sequence scaling')
      batch_size = common_layers.shape_list(logits)[0]
      features['sequence_scale'] = tf.ones([batch_size, 1], dtype=tf.float32)
    tile_num = self.hparams.greedy_sample_count
    sample_inputs = self.tile_with_adjacent_repeats(features['inputs'], tile_num)
    samples = self.sample_for_mrt(logits, features, tile_num, sample_inputs)
    scales = tf.tile(features['sequence_scale'], [tile_num, 1])
    if self.hparams.mrt_use_batch_bleu:
      scales = tf.py_func(self.batch_bleus, [samples, features['targets']], tf.float32)
    elif self.hparams.mrt_use_gleu:
      scales = tf.py_func(self.batch_bleus, [samples, features['targets'], features['inputs']], tf.float32)
      scales = tf.expand_dims(scales, -1)
    elif self.hparams.mrt_use_ter:
      scales = tf.py_func(self.get_ters, [samples, scales, features['targets']], tf.float32)
    elif self.hparams.mrt_equal_scaling:
      scale = self.hparams.mrt_scale_factors
      if self.hparams.mrt_scale_factors_by_sample_count:
        scale /= (self.hparams.greedy_sample_count - 1)
      scales *= scale
    else:
      scales = tf.py_func(self.seq_bleus, [samples, scales, features['targets']], tf.float32)
    self.set_features_samples_and_inputs(features, scales, samples, sample_inputs)


  def set_features_samples_and_inputs(self, features, scales, samples, sample_inputs):
    if self.hparams.mrt_include_gold:
      tf.logging.info('Appending reference to samples')
      scales = tf.concat([features['sequence_scale'], scales], axis=0)
      samples = tf.concat([features['targets'], samples], axis=0)
      sample_inputs = tf.concat([features['inputs'], sample_inputs], axis=0)
    features['sequence_scale'] = scales
    features['targets'] = samples
    features['inputs'] = sample_inputs

  def mix_gold_sampled(self, gold_targets, sampled_targets):
    return tf.where(
        tf.less(
            tf.random_uniform(common_layers.shape_list(sampled_targets)),
            self.hparams.mrt_gold_mixin_prob), gold_targets,
        sampled_targets)

  def sample_for_mrt(self, logits, features, tile_num, sample_inputs):
    original_targets = features['targets']
    original_inputs = features['inputs']
    target_shape = common_layers.shape_list(original_targets)
    if self.hparams.mrt_autoregressive_sample or self.hparams.mrt_beam_sample:
      self.set_mode(tf.estimator.ModeKeys.PREDICT)
      features['targets'] = None
      if self.hparams.mrt_autoregressive_sample:
        features['inputs'] = sample_inputs
        samples = self._fast_decode_nodp(
          features, decode_length=target_shape[1], beam_size=1, top_beams=1, alpha=self.hparams.mrt_beam_alpha)
        samples = tf.expand_dims(tf.expand_dims(samples['outputs'], -1), -1)
      else:
        samples = self._fast_decode_nodp(
          features, decode_length=target_shape[1], beam_size=tile_num, top_beams=tile_num, alpha=self.hparams.mrt_beam_alpha)
        samples = samples['outputs']
        #samples += tf.cast(do_hacky_print(samples), tf.int32)
        sample_shape = common_layers.shape_list(samples)
        samples = tf.reshape(samples, [sample_shape[0] * tile_num, sample_shape[2], 1, 1])
      self.set_mode(tf.estimator.ModeKeys.TRAIN)
      #samples = self.tile_with_adjacent_repeats(samples, tile_num)
    else:
      sample_logits = self.tile_with_adjacent_repeats(logits, tile_num)
      samples = common_layers.sample_with_temperature(sample_logits, self.hparams.sampling_temp)
    samples = tf.to_int32(samples)
    gold_targets = self.tile_with_adjacent_repeats(original_targets, tile_num)
    if self.hparams.mrt_gold_mixin_prob > 0.0:
      samples = self.mix_gold_sampled(gold_targets, samples)
    target_shape[0] *= tile_num
    samples = tf.py_func(self.mask_samples, [samples, gold_targets], tf.int32)
    #samples += tf.cast(do_hacky_print(samples), tf.int32)
    samples = tf.reshape(samples, target_shape)
    features['targets'] = original_targets
    features['inputs'] = original_inputs
    return samples


  def get_bleu_from_matches(self, hyps_by_order, matches_by_order):
    smooth = 1.0
    bleu_smooth = self.hparams.mrt_scale_smoothing
    precisions = self.get_empty_max_order()
    if not sum(hyps_by_order):
      return 0.0
    for i in xrange(self.hparams.mrt_bleu_max_order):
      if hyps_by_order[i]:
        if matches_by_order[i]:
          precisions[i] = matches_by_order[i] / hyps_by_order[i]
        elif bleu_smooth:
          smooth *= 2
          precisions[i] = 1.0 / (smooth * hyps_by_order[i])
    bleu = math.exp(sum(math.log(p) for p in precisions if p) / self.hparams.mrt_bleu_max_order)
    return bleu

  def get_sentence_bleu(self, ref_ngrams, ref_len, hyp_ngrams, hyp_len):
    matches_by_order = self.get_empty_max_order()
    hyp_ngrams_by_order = self.get_empty_max_order()
    self.ngram_matches(hyp_ngrams, ref_ngrams, matches_by_order, hyp_ngrams_by_order)
    bleu = self.get_bleu_from_matches(hyp_ngrams_by_order, matches_by_order)
    if self.hparams.mrt_use_brevity_penalty:
      bleu *= self.get_bleu_bp(hyp_len, ref_len)
    return bleu


  def get_ters(self, samples, scales, targets):
    for idx in range(len(targets)):
      ref = decoding._save_until_eos(targets[idx], is_image=False)
      for sample_idx in range(idx * self.hparams.greedy_sample_count,
                              (idx + 1) * self.hparams.greedy_sample_count):
        sample = decoding._save_until_eos(samples[sample_idx], is_image=False)
        scales[sample_idx] = pyter.ter(sample, ref)
    scales = self._maybe_adjust_scales(scales)
    return np.asarray(scales, dtype=np.float32)


  def get_bleu_bp(self, hyp_len, ref_len):
    if ref_len:
      ratio = max(hyp_len, 1) / ref_len
      bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
    else:
      bp = 1.0
    return bp

  def seq_bleus(self, samples, scales, targets):
    for idx in range(len(targets)):
      ref_ngrams, ref_len = self.get_trimmed_ngrams_and_len(targets[idx])
      for sample_idx in range(idx * self.hparams.greedy_sample_count,
                              (idx + 1) * self.hparams.greedy_sample_count):
        sample_ngrams, sample_len = self.get_trimmed_ngrams_and_len(samples[sample_idx])
        bleu = self.get_sentence_bleu(ref_ngrams, ref_len, sample_ngrams, sample_len)
        scales[sample_idx] = bleu
    scales = self._maybe_adjust_scales(scales)
    return np.asarray(scales, dtype=np.float32)

  def empty_max_order_per_set(self):
    return [self.get_empty_max_order() for _ in range(self.hparams.greedy_sample_count)]

  def get_empty_max_order(self):
    return self.hparams.mrt_bleu_max_order * [0]

  def get_trimmed_ngrams_and_len(self, seq):
    trimmed_seq = decoding._save_until_eos(seq, is_image=False)
    ngrams = bleu_hook._get_ngrams(trimmed_seq, self.hparams.mrt_bleu_max_order)
    return ngrams, len(trimmed_seq)

  def get_ordered_batch_bleus(self, samples, targets, inputs, hyps, matches, lens, sample_set_orders):
    all_ref_len = 0
    for target_idx in range(len(targets)):
      ref_ngrams, ref_len = self.get_trimmed_ngrams_and_len(targets[target_idx])
      all_ref_len += ref_len
      if inputs is not None:
        source_ngrams, _ = self.get_trimmed_ngrams_and_len(inputs[target_idx])
        not_src_ngrams = self.get_not_src_ngrams(ref_ngrams, source_ngrams)
      hyp_matches = self.empty_max_order_per_set()
      hyp_hyps = self.empty_max_order_per_set()
      hyp_lens = []
      for idx in range(self.hparams.greedy_sample_count):
        untrimmed_sample = samples[target_idx * self.hparams.greedy_sample_count + idx]
        hyp_ngrams, hyp_len = self.get_trimmed_ngrams_and_len(untrimmed_sample)
        if self.hparams.mrt_order_by_matches:
          self.ngram_matches(hyp_ngrams, ref_ngrams, hyp_matches[idx], hyp_hyps[idx])
          hyp_lens.append(hyp_len)
          if inputs is not None:
            self.not_src_ngram_matches(hyp_ngrams, not_src_ngrams, hyp_matches[idx], hyp_hyps[idx])
        else:
          self.ngram_matches(hyp_ngrams, ref_ngrams, matches[idx], hyps[idx])
          if inputs is not None:
            self.not_src_ngram_matches(hyp_ngrams, not_src_ngrams, matches[idx], hyps[idx])
          lens[idx] += hyp_len
      if self.hparams.mrt_order_by_matches:
        total_matches = [sum(m) for m in hyp_matches]
        for set_idx, (ordered_idx, _) in enumerate(
            sorted(enumerate(total_matches), key=lambda x: x[1])):
          sample_set_orders[target_idx].append(ordered_idx)
          lens[set_idx] += hyp_lens[ordered_idx]
          for order in range(self.hparams.mrt_bleu_max_order):
            matches[set_idx][order] += hyp_matches[ordered_idx][order]
            hyps[set_idx][order] += hyp_hyps[ordered_idx][order]
    return all_ref_len


  def batch_bleus(self, samples, targets, inputs=None):
    """
    If we have N samples per source sentence, take the N-wise sample sets and find the average set bleu metric
    We need two quantities for our metric:
    - The average set bleu across all N possible sets
    - The set bleu for each possible set
    !! this makes a lot more sense if the samples are ordered somehow !!
    """
    set_hyps = self.empty_max_order_per_set()
    set_matches = self.empty_max_order_per_set()
    set_lens = {idx: 0 for idx in range(self.hparams.greedy_sample_count)}
    set_sample_orders = [[] for _ in range(len(targets))] # in order, for each set, id of samples corresponding to the best set, 2nd best etc
    ref_len = self.get_ordered_batch_bleus(samples, targets, inputs, set_hyps, set_matches, set_lens, set_sample_orders)
    set_bleus = []
    for set_idx in range(self.hparams.greedy_sample_count):
      set_bleu = self.get_bleu_from_matches(set_hyps[set_idx], set_matches[set_idx])
      if self.hparams.mrt_use_brevity_penalty:
        set_bleu *= self.get_bleu_bp(set_lens[set_idx], ref_len)
      set_bleus.append(set_bleu)
    if self.hparams.mrt_include_gold_in_av:
      set_bleus.append(1.0)
      set_bleus = self._maybe_adjust_scales(np.asarray(set_bleus, dtype=np.float32))
      set_bleus = set_bleus[:-1]
    else:
      set_bleus = self._maybe_adjust_scales(np.asarray(set_bleus, dtype=np.float32))
    if self.hparams.mrt_order_by_matches:
      scales = []
      for sample_order in set_sample_orders:
        scales.append(set_bleus[np.asarray(sample_order, dtype=np.int32)])
      scales = np.asarray(scales)
    else:
      scales = np.tile(set_bleus, len(targets))
    scales = scales.reshape([scales.size, 1])
    return scales


  def _maybe_adjust_scales(self, scales):
    if self.hparams.mrt_use_negative_bleu:
      scales = -scales
    elif self.hparams.mrt_zero_bleu_high and not self.hparams.mrt_use_ter:
      scales = 1.0 - scales
    if self.hparams.mrt_subtract_av_metric:
      scales -= self.hparams.mrt_scale_var_reduce * np.mean(scales)
    if self.hparams.mrt_floor_loss_to_zero:
      scales = scales.clip(min=0.0)
    tf.logging.info(scales.squeeze())
    scales *= self.hparams.mrt_scale_factors
    if self.hparams.mrt_scale_factors_by_sample_count:
      scales /= (self.hparams.greedy_sample_count - 1)
    return scales


  def ngram_matches(self, hyp_ngrams, ref_ngrams, matches, hyps):
    for ngram, count in hyp_ngrams.items():
      hyps[len(ngram) - 1] += count
    for ngram, count in ref_ngrams.items():
      if hyp_ngrams[ngram] > 0:
        matches[len(ngram) - 1] += min(count, hyp_ngrams[ngram])

  def get_not_src_ngrams(self, ref_ngrams, source_ngrams):
    ref_not_source = collections.Counter()
    for ngram, count in ref_ngrams.items():
      if count - source_ngrams[ngram] > 0:
        ref_not_source[ngram] = count - source_ngrams[ngram]
    return ref_not_source

  def not_src_ngram_matches(self, hyp_ngrams, ref_not_src_ngrams, matches, hyps):
    for ngram, count in ref_not_src_ngrams.items():
      if hyp_ngrams[ngram] > 0:
        matches[len(ngram) - 1] += min(count, hyp_ngrams[ngram])
      hyps[len(ngram) - 1] += count


  def model_fn(self, features):
    output, losses = self.forward_pass_features(features)
    logits = self.top(output, features)
    
    do_sample = (self.hparams.mode == tf.estimator.ModeKeys.TRAIN and self.hparams.do_mrt)
    if do_sample:
      tf.logging.info('Sampling')
      self.get_features_samples_and_inputs(features, logits)
      output, losses = self.forward_pass_features(features)
      #losses["training"] = do_hacky_print(features['targets']) + do_hacky_print(features['inputs'])
      second_pass_logits = self.top(output, features)
      #losses["training"] += self.loss(second_pass_logits, features)
      losses["training"] = self.loss(second_pass_logits, features)
    else:
      losses["training"] = self.loss(logits, features)
    return logits, losses


@registry.register_model
class TransformerDiscriminative(Transformer):
  """Transformer with discriminative training."""
  pass

def hacky_print(t):
  tf.logging.info(np.squeeze(t))
  return np.float32(0.0)

def do_hacky_print(tensor_to_log):
  unchanged = tf.py_func(hacky_print, [tensor_to_log], tf.float32)
  return tf.reshape(0.0 * tf.reduce_sum(unchanged), ())

def fast_decode(encoder_output,
                encoder_decoder_attention_bias,
                symbols_to_logits_fn,
                hparams,
                decode_length,
                vocab_size,
                beam_size=1,
                top_beams=1,
                alpha=1.0,
                eos_id=beam_search.EOS_ID):
  """Given encoder output and a symbols to logits function, does fast decoding.

  Implements both greedy and beam search decoding, uses beam search iff
  beam_size > 1, otherwise beam search related arguments are ignored.

  Args:
    encoder_output: Output from encoder.
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
    symbols_to_logits_fn: Incremental decoding; function mapping triple
      `(ids, step, cache)` to symbol logits.
    hparams: run hyperparameters
    decode_length: an integer.  How many additional timesteps to decode.
    vocab_size: Output vocabulary size.
    beam_size: number of beams.
    top_beams: an integer. How many of the beams to return.
    alpha: Float that controls the length penalty. larger the alpha, stronger
      the preference for slonger translations.
    eos_id: End-of-sequence symbol in beam search.

  Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if top_beams == 1 or
              [batch_size, top_beams, <= decode_length] otherwise
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
  """
  batch_size = common_layers.shape_list(encoder_output)[0]
  key_channels = hparams.attention_key_channels or hparams.hidden_size
  value_channels = hparams.attention_value_channels or hparams.hidden_size
  num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers

  cache = {
      "layer_%d" % layer: {
          "k": tf.zeros([batch_size, 0, key_channels]),
          "v": tf.zeros([batch_size, 0, value_channels]),
      }
      for layer in range(num_layers)
  }

  cache["encoder_output"] = encoder_output
  cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

  if beam_size > 1:  # Beam Search
    tf.logging.info('Beginning beam search')
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)
    decoded_ids, scores = beam_search.beam_search(
        symbols_to_logits_fn,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        alpha,
        states=cache,
        eos_id=eos_id,
        stop_early=(top_beams == 1))

    if top_beams == 1:
      decoded_ids = decoded_ids[:, 0, 1:]
    else:
      decoded_ids = decoded_ids[:, :top_beams, 1:]
  else:  # Greedy
    tf.logging.info('Doing greedy decoding')
    def inner_loop(i, finished, next_id, decoded_ids, cache):
      logits, cache = symbols_to_logits_fn(next_id, i, cache)
      temperature = (0.0 if hparams.sampling_method == "argmax" else
                     hparams.sampling_temp)
      next_id = common_layers.sample_with_temperature(logits, temperature)
      finished |= tf.equal(next_id, eos_id)
      next_id = tf.expand_dims(next_id, axis=1)
      decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
      return i + 1, finished, next_id, decoded_ids, cache

    def is_not_finished(i, finished, *_):
      return (i < decode_length) & tf.logical_not(tf.reduce_all(finished))

    decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
    finished = tf.zeros([batch_size], tf.bool)
    next_id = tf.zeros([batch_size, 1], dtype=tf.int64)
    _, _, _, decoded_ids, _ = tf.while_loop(
        is_not_finished,
        inner_loop,
        [tf.constant(0), finished, next_id, decoded_ids, cache],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            nest.map_structure(beam_search.get_state_shape_invariants, cache),
        ],
      back_prop=False)
    scores = None
  return {"outputs": decoded_ids, "scores": scores}


@registry.register_model
class TransformerEncoder(t2t_model.T2TModel):
  """Transformer, encoder only."""

  def body(self, features):
    hparams = self._hparams
    inputs = features["inputs"]
    target_space = features["target_space_id"]

    inputs = common_layers.flatten4d3d(inputs)

    (encoder_input, encoder_self_attention_bias, _) = (
        transformer_prepare_encoder(inputs, target_space, hparams))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)
    encoder_output = transformer_encoder(
        encoder_input, encoder_self_attention_bias, hparams,
        nonpadding=features_to_nonpadding(features, "inputs"))
    encoder_output = tf.expand_dims(encoder_output, 2)

    return encoder_output


def features_to_nonpadding(features, inputs_or_targets="inputs"):
  key = inputs_or_targets + "_segmentation"
  if features and key in features:
    return tf.minimum(features[key], 1.0)
  return None


def transformer_prepare_encoder(inputs, target_space, hparams, features=None):
  """Prepare one shard of the model for the encoder.

  Args:
    inputs: a Tensor.
    target_space: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well.
      This is needed now for "packed" datasets.

  Returns:
    encoder_input: a Tensor, bottom of encoder stack
    encoder_self_attention_bias: a bias tensor for use in encoder self-attention
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
  """
  ishape_static = inputs.shape.as_list()
  encoder_input = inputs
  if features and "inputs_segmentation" in features:
    # Packed dataset.  Keep the examples from seeing each other.
    inputs_segmentation = features["inputs_segmentation"]
    inputs_position = features["inputs_position"]
    targets_segmentation = features["targets_segmentation"]
    encoder_self_attention_bias = common_attention.attention_bias_same_segment(
        inputs_segmentation, inputs_segmentation)
    encoder_decoder_attention_bias = (
        common_attention.attention_bias_same_segment(
            targets_segmentation, inputs_segmentation))
  else:
    # Usual case - not a packed dataset.
    encoder_padding = common_attention.embedding_to_padding(encoder_input)
    ignore_padding = common_attention.attention_bias_ignore_padding(
        encoder_padding)
    encoder_self_attention_bias = ignore_padding
    encoder_decoder_attention_bias = ignore_padding
    inputs_position = None
  if hparams.proximity_bias:
    encoder_self_attention_bias += common_attention.attention_bias_proximal(
        common_layers.shape_list(inputs)[1])
  # Append target_space_id embedding to inputs.
  if target_space is not None:
    emb_target_space = common_layers.embedding(
      target_space, 32, ishape_static[-1], name="target_space_embedding")
    emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
    encoder_input += emb_target_space
  if hparams.pos == "timing":
    if inputs_position is not None:
      encoder_input = common_attention.add_timing_signal_1d_given_position(
          encoder_input, inputs_position)
    else:
      encoder_input = common_attention.add_timing_signal_1d(encoder_input)
  return (encoder_input, encoder_self_attention_bias,
          encoder_decoder_attention_bias)


def transformer_prepare_decoder(targets, hparams, features=None):
  """Prepare one shard of the model for the decoder.

  Args:
    targets: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well.
      This is needed now for "packed" datasets.

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a bias tensor for use in encoder self-attention
  """
  if hparams.prepend_mode == "prepend_inputs_full_attention":
    decoder_self_attention_bias = (
        common_attention.attention_bias_prepended(
            common_attention.embedding_to_padding(targets)))
  else:
    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(
            common_layers.shape_list(targets)[1]))

  if features and "targets_segmentation" in features:
    # "Packed" dataset - keep the examples from seeing each other.
    targets_segmentation = features["targets_segmentation"]
    targets_position = features["targets_position"]
    decoder_self_attention_bias += common_attention.attention_bias_same_segment(
        targets_segmentation, targets_segmentation)
  else:
    targets_position = None
  if hparams.proximity_bias:
    decoder_self_attention_bias += common_attention.attention_bias_proximal(
        common_layers.shape_list(targets)[1])
  decoder_input = common_layers.shift_right_3d(targets)
  if hparams.pos == "timing":
    if targets_position is not None:
      decoder_input = common_attention.add_timing_signal_1d_given_position(
          decoder_input, targets_position)
    else:
      decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  return (decoder_input, decoder_self_attention_bias)


def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True):
  """A stack of transformer layers.

  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This must either be
      passed in, which we do for "packed" datasets, or inferred from
      encoder_self_attention_bias.  The knowledge about padding is used
      for pad_remover(efficiency) and to mask out padding in convoltutional
      layers.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.

  Returns:
    y: a Tensors
  """
  x = encoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  with tf.variable_scope(name):
    if nonpadding is not None:
      padding = 1.0 - nonpadding
    else:
      padding = common_attention.attention_bias_to_padding(
          encoder_self_attention_bias)
      nonpadding = 1.0 - padding
    pad_remover = None
    if hparams.use_pad_remover and not common_layers.is_on_tpu():
      pad_remover = expert_utils.PadRemover(padding)
    for layer in xrange(hparams.num_encoder_layers or
                        hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              encoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              save_weights_to=save_weights_to,
              max_relative_position=hparams.max_relative_position,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims)
          x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams), hparams, pad_remover,
              conv_padding="SAME", nonpadding_mask=nonpadding)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it shuold also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)


def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        name="decoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True):
  """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention
      (see common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used
      to mask out padding in convoltutional layers.  We generally only
      need this mask for "packed" datasets, because for ordinary datasets,
      no padding is ever followed by nonpadding.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.

  Returns:
    y: a Tensors
  """
  x = decoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  with tf.variable_scope(name):
    for layer in xrange(hparams.num_decoder_layers or
                        hparams.num_hidden_layers):
      layer_name = "layer_%d" % layer
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope(layer_name):
        with tf.variable_scope("self_attention"):
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              save_weights_to=save_weights_to,
              max_relative_position=hparams.max_relative_position,
              cache=layer_cache,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims)
          x = common_layers.layer_postprocess(x, y, hparams)
        if encoder_output is not None:
          with tf.variable_scope("encdec_attention"):
            # TODO(llion): Add caching.
            y = common_attention.multihead_attention(
                common_layers.layer_preprocess(x, hparams),
                encoder_output,
                encoder_decoder_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                save_weights_to=save_weights_to,
                make_image_summary=make_image_summary,
                dropout_broadcast_dims=attention_dropout_broadcast_dims)
            x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams), hparams,
              conv_padding="LEFT", nonpadding_mask=nonpadding)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it shuold also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)


def transformer_ffn_layer(x,
                          hparams,
                          pad_remover=None,
                          conv_padding="LEFT",
                          nonpadding_mask=None):
  """Feed-forward layer in the transformer.

  Args:
    x: a Tensor of shape [batch_size, length, hparams.hidden_size]
    hparams: hyperparmeters for model
    pad_remover: an expert_utils.PadRemover object tracking the padding
      positions. If provided, when using convolutional settings, the padding
      is removed before applying the convolution, and restored afterward. This
      can give a significant speedup.
    conv_padding: a string - either "LEFT" or "SAME".
    nonpadding_mask: an optional Tensor with shape [batch_size, length].
      needed for convolutoinal layers with "SAME" padding.
      Contains 1.0 in positions corresponding to nonpadding.

  Returns:
    a Tensor of shape [batch_size, length, hparams.hidden_size]
  """
  ffn_layer = hparams.ffn_layer
  relu_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "relu_dropout_broadcast_dims", "")))
  if ffn_layer == "conv_hidden_relu":
    # Backwards compatibility
    ffn_layer = "dense_relu_dense"
  if ffn_layer == "dense_relu_dense":
    # In simple convolution mode, use `pad_remover` to speed up processing.
    if pad_remover:
      original_shape = common_layers.shape_list(x)
      # Collapse `x` across examples, and remove padding positions.
      x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
      x = tf.expand_dims(pad_remover.remove(x), axis=0)
    conv_output = common_layers.dense_relu_dense(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        dropout=hparams.relu_dropout,
        dropout_broadcast_dims=relu_dropout_broadcast_dims)
    if pad_remover:
      # Restore `conv_output` to the original shape of `x`, including padding.
      conv_output = tf.reshape(
          pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
    return conv_output
  elif ffn_layer == "conv_relu_conv":
    return common_layers.conv_relu_conv(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        first_kernel_size=3,
        second_kernel_size=1,
        padding=conv_padding,
        nonpadding_mask=nonpadding_mask,
        dropout=hparams.relu_dropout)
  elif ffn_layer == "parameter_attention":
    return common_attention.parameter_attention(
        x, hparams.parameter_attention_key_channels or hparams.hidden_size,
        hparams.parameter_attention_value_channels or hparams.hidden_size,
        hparams.hidden_size, hparams.filter_size, hparams.num_heads,
        hparams.attention_dropout)
  elif ffn_layer == "conv_hidden_relu_with_sepconv":
    return common_layers.conv_hidden_relu(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        kernel_size=(3, 1),
        second_kernel_size=(31, 1),
        padding="LEFT",
        dropout=hparams.relu_dropout)
  else:
    assert ffn_layer == "none"
    return x


@registry.register_hparams
def transformer_base_v1():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.norm_type = "layer"
  hparams.hidden_size = 512
  hparams.batch_size = 4096
  hparams.max_length = 256
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 1.0
  hparams.num_hidden_layers = 6
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.num_sampled_classes = 0
  hparams.label_smoothing = 0.1
  hparams.shared_embedding_and_softmax_weights = True
  hparams.symbol_modality_num_shards = 16

  # Add new ones like this.
  hparams.add_hparam("filter_size", 2048)
  # Layer-related flags. If zero, these fall back on hparams.num_hidden_layers.
  hparams.add_hparam("num_encoder_layers", 0)
  hparams.add_hparam("num_decoder_layers", 0)
  # Attention-related flags.
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("ffn_layer", "dense_relu_dense")
  hparams.add_hparam("parameter_attention_key_channels", 0)
  hparams.add_hparam("parameter_attention_value_channels", 0)
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("attention_dropout_broadcast_dims", "")
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("relu_dropout_broadcast_dims", "")
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("nbr_decoder_problems", 1)
  hparams.add_hparam("proximity_bias", False)
  hparams.add_hparam("use_pad_remover", True)
  hparams.add_hparam("self_attention_type", "dot_product")
  hparams.add_hparam("max_relative_position", 0)
  return hparams


@registry.register_hparams
def transformer_base_v2():
  """Set of hyperparameters."""
  hparams = transformer_base_v1()
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate = 0.2
  return hparams


@registry.register_hparams
def transformer_base():
  # Update parameters here, then occasionally cut a versioned set, e.g.
  # transformer_base_v2.
  hparams = transformer_base_v2()
  hparams.optimizer_adam_beta2 = 0.997
  return hparams


@registry.register_hparams
def transformer_big():
  """HParams for transfomer big model on WMT."""
  hparams = transformer_base()
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.num_heads = 16
  hparams.layer_prepostprocess_dropout = 0.3
  return hparams


@registry.register_hparams
def transformer_big_single_gpu():
  """HParams for transformer big model for single gpu."""
  hparams = transformer_big()
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.learning_rate_warmup_steps = 16000
  return hparams


@registry.register_hparams
def transformer_base_single_gpu():
  """HParams for transformer base model for single gpu."""
  hparams = transformer_base()
  hparams.batch_size = 2048
  hparams.learning_rate_warmup_steps = 16000
  return hparams


@registry.register_hparams
def transformer_parsing_base():
  """Hparams for parsing on wsj only."""
  hparams = transformer_base()
  hparams.attention_dropout = 0.2
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.max_length = 512
  hparams.learning_rate_warmup_steps = 16000
  hparams.hidden_size = 1024
  hparams.learning_rate = 0.05
  hparams.shared_embedding_and_softmax_weights = False
  return hparams


@registry.register_hparams
def transformer_parsing_big():
  """HParams for parsing on wsj semi-supervised."""
  hparams = transformer_big()
  hparams.max_length = 512
  hparams.shared_source_target_embedding = False
  hparams.learning_rate_warmup_steps = 4000
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.batch_size = 2048
  hparams.learning_rate = 0.05
  return hparams


@registry.register_hparams
def transformer_parsing_ice():
  """Hparams for parsing and tagging Icelandic text."""
  hparams = transformer_base_single_gpu()
  hparams.batch_size = 4096
  hparams.shared_embedding_and_softmax_weights = False
  return hparams


@registry.register_hparams
def transformer_tiny():
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 128
  hparams.filter_size = 512
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def transformer_test():
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 16
  hparams.filter_size = 8
  hparams.num_heads = 2
  return hparams


@registry.register_hparams
def transformer_small():
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 256
  hparams.filter_size = 1024
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def transformer_l2():
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  return hparams


@registry.register_hparams
def transformer_l4():
  hparams = transformer_base()
  hparams.num_hidden_layers = 4
  return hparams


@registry.register_hparams
def transformer_l8():
  hparams = transformer_base()
  hparams.num_hidden_layers = 8
  return hparams


@registry.register_hparams
def transformer_l10():
  hparams = transformer_base()
  hparams.num_hidden_layers = 10
  return hparams


@registry.register_hparams
def transformer_h1():
  hparams = transformer_base()
  hparams.num_heads = 1
  return hparams


@registry.register_hparams
def transformer_h4():
  hparams = transformer_base()
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def transformer_h16():
  hparams = transformer_base()
  hparams.num_heads = 16
  return hparams


@registry.register_hparams
def transformer_h32():
  hparams = transformer_base()
  hparams.num_heads = 32
  return hparams


@registry.register_hparams
def transformer_k128():
  hparams = transformer_base()
  hparams.attention_key_channels = 128
  return hparams


@registry.register_hparams
def transformer_k256():
  hparams = transformer_base()
  hparams.attention_key_channels = 256
  return hparams


@registry.register_hparams
def transformer_ff1024():
  hparams = transformer_base()
  hparams.filter_size = 1024
  return hparams


@registry.register_hparams
def transformer_ff4096():
  hparams = transformer_base()
  hparams.filter_size = 4096
  return hparams


@registry.register_hparams
def transformer_dr0():
  hparams = transformer_base()
  hparams.layer_prepostprocess_dropout = 0.0
  return hparams


@registry.register_hparams
def transformer_dr2():
  hparams = transformer_base()
  hparams.layer_prepostprocess_dropout = 0.2
  return hparams


@registry.register_hparams
def transformer_ls0():
  hparams = transformer_base()
  hparams.label_smoothing = 0.0
  return hparams


@registry.register_hparams
def transformer_ls2():
  hparams = transformer_base()
  hparams.label_smoothing = 0.2
  return hparams


@registry.register_hparams
def transformer_hs256():
  hparams = transformer_base()
  hparams.hidden_size = 256
  return hparams


@registry.register_hparams
def transformer_hs1024():
  hparams = transformer_base()
  hparams.hidden_size = 1024
  return hparams


@registry.register_hparams
def transformer_big_dr1():
  hparams = transformer_base()
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.num_heads = 16
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def transformer_big_enfr():
  hparams = transformer_big_dr1()
  hparams.shared_embedding_and_softmax_weights = False
  hparams.filter_size = 8192
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def transformer_big_dr2():
  hparams = transformer_big_dr1()
  hparams.layer_prepostprocess_dropout = 0.2
  return hparams


@registry.register_hparams
def transformer_parameter_attention_a():
  hparams = transformer_base()
  hparams.ffn_layer = "parameter_attention"
  hparams.filter_size = 1536
  return hparams


@registry.register_hparams
def transformer_parameter_attention_b():
  hparams = transformer_base()
  hparams.ffn_layer = "parameter_attention"
  hparams.filter_size = 512
  hparams.parameter_attention_key_channels = 1024
  hparams.parameter_attention_value_channels = 1024
  hparams.num_heads = 16
  return hparams


@registry.register_hparams
def transformer_prepend_v2():
  hparams = transformer_base_v2()
  hparams.prepend_mode = "prepend_inputs_masked_attention"
  hparams.max_length = 0
  return hparams


@registry.register_hparams
def transformer_prepend_v1():
  hparams = transformer_base_v1()
  hparams.prepend_mode = "prepend_inputs_masked_attention"
  hparams.max_length = 0
  return hparams


@registry.register_hparams
def transformer_prepend():
  return transformer_prepend_v2()


@registry.register_ranged_hparams("transformer_base")
def transformer_base_range(rhp):
  """Small range of hyperparameters."""
  hparams = transformer_base()
  common_hparams.fill_ranged_hparams_from_hparams(hparams, rhp)
  # After starting from base, set intervals for some parameters.
  rhp.set_float("learning_rate", 0.3, 3.0, scale=rhp.LOG_SCALE)
  rhp.set_discrete("learning_rate_warmup_steps",
                   [1000, 2000, 4000, 8000, 16000])
  rhp.set_float("initializer_gain", 0.5, 2.0)
  rhp.set_float("optimizer_adam_beta1", 0.85, 0.95)
  rhp.set_float("optimizer_adam_beta2", 0.97, 0.99)
  rhp.set_float("weight_decay", 0.0, 2.0)


@registry.register_hparams
def transformer_relative():
  """Use relative position embeddings instead of absolute position encodings."""
  hparams = transformer_base()
  hparams.pos = None
  hparams.self_attention_type = "dot_product_relative"
  hparams.max_relative_position = 20
  return hparams


@registry.register_hparams
def transformer_relative_tiny():
  hparams = transformer_relative()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 128
  hparams.filter_size = 512
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def transformer_relative_big():
  hparams = transformer_big()
  hparams.pos = None
  hparams.self_attention_type = "dot_product_relative"
  hparams.max_relative_position = 20
  return hparams


def update_hparams_for_tpu(hparams):
  """Change hparams to be compatible with TPU training."""

  # Adafactor uses less memory than Adam.
  hparams.optimizer = "Adafactor"

  # Avoid an expensive concat on TPU.
  # >1 shards helps with faster parameter distribution on multi-GPU machines
  hparams.symbol_modality_num_shards = 1

  # Adaptive batch sizes and sequence lengths are not supported on TPU.
  # Instead, every batch has the same sequence length and the same batch size.
  # Longer sequences are dropped and shorter ones are padded.
  #
  # It is therefore suggested to use a problem where examples have been combined
  # to a longer length, e.g. the "_packed" problems.
  #
  # For problems with variable sequence lengths, this parameter controls the
  # maximum sequence length.  Shorter sequences are dropped and longer ones
  # are padded.
  #
  # For problems with fixed sequence lengths - e.g. the "_packed" problems,
  # this hyperparameter is ignored.
  hparams.max_length = 64

  # TPUs have less memory than GPUs, so decrease the batch size
  hparams.batch_size = 2048

  # Using noise broadcast in the dropout layers saves memory during training.
  hparams.attention_dropout_broadcast_dims = "0,1"  # batch, heads
  hparams.relu_dropout_broadcast_dims = "1"  # length
  hparams.layer_prepostprocess_dropout_broadcast_dims = "1"  # length


@registry.register_hparams
def transformer_tpu():
  """HParams for Transformer model on TPU."""
  hparams = transformer_base()
  update_hparams_for_tpu(hparams)
  return hparams


@registry.register_hparams
def transformer_packed_tpu():
  """Deprecated alias for transformer_tpu()."""
  return transformer_tpu()


@registry.register_hparams
def transformer_big_tpu():
  hparams = transformer_big()
  update_hparams_for_tpu(hparams)
  return hparams


@registry.register_hparams
def transformer_tiny_tpu():
  hparams = transformer_tiny()
  update_hparams_for_tpu(hparams)
  return hparams


@registry.register_ranged_hparams
def transformer_tiny_tpu_range(rhp):
  """Small range of hyperparameters."""
  hparams = transformer_tiny_tpu()
  common_hparams.fill_ranged_hparams_from_hparams(hparams, rhp)
  rhp.set_float("learning_rate", 0.3, 3.0, scale=rhp.LOG_SCALE)
  rhp.set_float("weight_decay", 0.0, 2.0)


@registry.register_ranged_hparams
def transformer_tpu_range(rhp):
  """Small range of hyperparameters."""
  hparams = transformer_tpu()
  common_hparams.fill_ranged_hparams_from_hparams(hparams, rhp)
  # After starting from base, set intervals for some parameters.
  rhp.set_float("learning_rate", 0.3, 3.0, scale=rhp.LOG_SCALE)
  rhp.set_discrete("learning_rate_warmup_steps",
                   [1000, 2000, 4000, 8000, 16000])
  rhp.set_float("initializer_gain", 0.5, 2.0)
  rhp.set_float("optimizer_adam_beta1", 0.85, 0.95)
  rhp.set_float("optimizer_adam_beta2", 0.97, 0.99)
  rhp.set_float("weight_decay", 0.0, 2.0)


@registry.register_ranged_hparams
def transformer_tpu_batch_range(rhp):
  hparams = transformer_tpu()
  common_hparams.fill_ranged_hparams_from_hparams(hparams, rhp)
  rhp.set_discrete("batch_size", [256, 512, 768, 1024])


@registry.register_hparams
def transformer_small_tpu():
  """TPU-friendly version of transformer_small.

  Returns:
    an hparams object.
  """
  hparams = transformer_small()
  update_hparams_for_tpu(hparams)
  return hparams


@registry.register_hparams
def transformer_clean():
  """No dropout, label smoothing, max_length."""
  hparams = transformer_base_v2()
  hparams.label_smoothing = 0.0
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.attention_dropout = 0.0
  hparams.relu_dropout = 0.0
  hparams.max_length = 0
  return hparams


@registry.register_hparams
def transformer_clean_big():
  hparams = transformer_clean()
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  return hparams


@registry.register_hparams
def transformer_clean_big_tpu():
  hparams = transformer_clean_big()
  update_hparams_for_tpu(hparams)
  return hparams


@registry.register_hparams
def transformer_tpu_with_conv():
  """Cut down on the number of heads, and use convs instead."""
  hparams = transformer_tpu()
  hparams.num_heads = 4   # heads are expensive on tpu
  hparams.ffn_layer = "conv_relu_conv"
  return hparams


@registry.register_hparams
def transformer_lm_tpu_0():
  """Hparams for training languagemodel_lm1b8k on tpu.  92M Params."""
  hparams = transformer_clean_big()
  update_hparams_for_tpu(hparams)
  hparams.num_heads = 4   # heads are expensive on tpu
  hparams.batch_size = 4096
  hparams.shared_embedding_and_softmax_weights = False
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def transformer_lm_tpu_1():
  """Hparams for training languagemodel_lm1b8k on tpu.  335M Params."""
  hparams = transformer_lm_tpu_0()
  hparams.hidden_size = 2048
  hparams.filter_size = 8192
  return hparams
