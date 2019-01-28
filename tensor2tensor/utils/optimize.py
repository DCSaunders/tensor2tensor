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

"""Optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

from tensor2tensor.utils import yellowfin
from tensor2tensor.utils import largebatch_optimizer

import tensorflow as tf

from tensorflow.python.framework import dtypes

from tensorflow.python.framework import ops
from tensor2tensor.utils.optimizer_wrappers import EWCOptimizer
from tensor2tensor.utils.optimizer_wrappers import ConditionalOptimizer


def optimize(loss, learning_rate, hparams, use_tpu=False):
  """Minimize loss."""
  loss = weight_decay_and_noise(loss, hparams, learning_rate)
  loss = tf.identity(loss, name="total_loss")
  log_variable_sizes(verbose=hparams.summarize_vars)
  diet_vars = [
      v for v in tf.global_variables() if v.dtype == dtypes.float16_ref
  ]
  log_variable_sizes(
      diet_vars, "Diet Variables", verbose=hparams.summarize_vars)
  if hparams.ewc_save_vars or hparams.ewc_load_vars:
    opt = EWCOptimizer(hparams.optimizer, learning_rate, hparams, use_tpu)
    if hparams.ewc_load_vars:
      loss += opt.get_ewc_loss()
  else:
    opt = ConditionalOptimizer(hparams.optimizer, learning_rate, hparams, use_tpu)
  if use_tpu:
    opt = tf.contrib.tpu.CrossShardOptimizer(opt)

  tf.summary.scalar("learning_rate", learning_rate)
  opt_summaries = ["loss", "global_gradient_norm"]
  if hparams.summarize_grads:
    tf.logging.info("Summarizing gradients")
    opt_summaries.extend(["gradients", "gradient_norm"])

  if hparams.clip_grad_norm:
    tf.logging.info("Clipping gradients, norm: %0.5f", hparams.clip_grad_norm)
  if hparams.grad_noise_scale:
    tf.logging.info("Adding noise to gradients, noise scale: %0.5f",
                    hparams.grad_noise_scale)

  train_op = tf.contrib.layers.optimize_loss(
      name="training",
      loss=loss,
      global_step=tf.train.get_or_create_global_step(),
      learning_rate=learning_rate,
      clip_gradients=hparams.clip_grad_norm or None,
      gradient_noise_scale=hparams.grad_noise_scale or None,
      optimizer=opt,
      summaries=opt_summaries,
      colocate_gradients_with_ops=True)
  return train_op


class ConditionalOptimizer(tf.train.Optimizer):
  """Conditional optimizer."""

  def __init__(self, optimizer_name, lr, hparams, use_tpu=False):
    if optimizer_name == "Adam" and use_tpu:
      # LazyAdamOptimizer does not work on TPU
      optimizer_name = "TrueAdam"

    tf.logging.info("Using optimizer %s", optimizer_name)

    if optimizer_name == "Adam":
      # We change the default epsilon for Adam and re-scale lr.
      # Using LazyAdam as it's much faster for large vocabulary embeddings.
      self._opt = tf.contrib.opt.LazyAdamOptimizer(
          lr / 500.0,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon)
    elif optimizer_name == "LargebatchAdam":
      self._opt = largebatch_optimizer.LargebatchAdamOptimizer(
          lr / 500.0,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon,
          n=hparams.largebatch_multiplier)
    elif optimizer_name == "Momentum":
      self._opt = tf.train.MomentumOptimizer(
          lr,
          momentum=hparams.optimizer_momentum_momentum,
          use_nesterov=hparams.optimizer_momentum_nesterov)
    elif optimizer_name == "YellowFin":
      self._opt = yellowfin.YellowFinOptimizer(
          learning_rate=lr, momentum=hparams.optimizer_momentum_momentum)
    elif optimizer_name == "TrueAdam":
      self._opt = tf.train.AdamOptimizer(
          lr / 500.0,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon)
    elif optimizer_name == "Adafactor":
      self._opt = AdafactorOptimizer(lr / 500.0)
    else:
      self._opt = tf.contrib.layers.OPTIMIZER_CLS_NAMES[optimizer_name](lr)

  def compute_gradients(self, loss, var_list=None, **kwargs):
    return self._opt.compute_gradients(loss, var_list, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    return self._opt.apply_gradients(
        grads_and_vars, global_step=global_step, name=name)


>>>>>>> dsaunders_v1.4.3_modified-accumulate_gradients
def _sqrt_decay(step):
  """Decay like 1 / sqrt(step), multiplied by 500 to normalize."""
  return 500.0 / tf.sqrt(tf.maximum(step, 1.0))


def _exp_decay_after(step, rate, from_which_step):
  """Decay exponentially by rate (per step) starting at from_which_step."""
  return tf.cond(
      step < from_which_step,
      lambda: tf.constant(1.0),
      lambda: rate**(step - from_which_step),
      name="exponential_decay_step_cond")


def piecewise_learning_rate(step, boundaries, values):
  """Scale learning rate according to the given schedule.

  Multipliers are not cumulative.

  Args:
    step: global step
    boundaries: List of steps to transition on.
    values: Multiplier to apply at each boundary transition.

  Returns:
    Scaled value for the learning rate.
  """
  values = [1.0] + values
  return tf.train.piecewise_constant(
      step, boundaries, values, name="piecewise_lr")

def get_global_step(hparams):
  global_step = tf.to_float(tf.train.get_or_create_global_step())
  if hparams is not None:
    try:
      if hparams.largebatch_multiplier > 1:
        tf.logging.info("Scaling down global step for optimizer by "
                        "largebatch_multiplier=%d" % hparams.largebatch_multiplier)
        largebatch_multiplier = tf.constant(hparams.largebatch_multiplier,
                                          dtype=tf.float32)
        global_step = global_step / largebatch_multiplier
    except AttributeError:
      pass
  return global_step


def learning_rate_decay(hparams, warmup_steps=0, num_worker_replicas=1, num_train_steps=1):
  """Inverse-decay learning rate until warmup_steps, then decay."""
  scheme = hparams.learning_rate_decay_scheme
  warmup_steps = tf.to_float(warmup_steps * num_worker_replicas)
  global_step = get_global_step(hparams)

  if not scheme or scheme == "none":
    return tf.constant(1.)

  tf.logging.info("Applying learning rate decay: %s.", scheme)

  if scheme == "exp":
    decay_steps = hparams.learning_rate_decay_steps
    p = (global_step - warmup_steps) / decay_steps
    if hparams.learning_rate_decay_staircase:
      p = tf.floor(p)
    return tf.pow(hparams.learning_rate_decay_rate, p)

  if scheme == "piecewise":
    return piecewise_learning_rate(global_step,
                                   hparams.learning_rate_boundaries,
                                   hparams.learning_rate_multiples)
    
  if scheme == "noam":
    return 5000.0 * hparams.hidden_size**-0.5 * tf.minimum(
        (global_step + 1) * warmup_steps**-1.5, (global_step + 1)**-0.5)

  if scheme == "cosine":
    cycle_steps = hparams.learning_rate_cosine_cycle_steps
    cycle_position = global_step % (2 * cycle_steps)
    cycle_position = cycle_steps - tf.abs(cycle_steps - cycle_position)
    return 0.5 * (1 + tf.cos(np.pi * cycle_position / cycle_steps))

  if scheme == "cyclelinear10x":
    # Cycle the rate linearly by 10x every warmup_steps, up and down.
    cycle_steps = warmup_steps
    cycle_position = global_step % (2 * cycle_steps)
    cycle_position = tf.to_float(  # Normalize to the interval [-1, 1].
        cycle_position - cycle_steps) / float(cycle_steps)
    cycle_position = 1.0 - tf.abs(cycle_position)  # 0 to 1 and back to 0.
    return (cycle_position + 0.1) * 3.0  # 10x difference each cycle (0.3-3).

  if scheme == "sqrt":
    return _sqrt_decay(global_step - warmup_steps)

  raise ValueError("Unrecognized learning rate decay scheme: %s" %
                   hparams.learning_rate_decay_scheme)


def learning_rate_warmup(warmup_steps, warmup_schedule="exp", hparams=None):
  """Learning rate warmup multiplier."""
  if not warmup_steps:
    return tf.constant(1.)

  tf.logging.info("Applying %s learning rate warmup for %d steps",
                  warmup_schedule, warmup_steps)

  warmup_steps = tf.to_float(warmup_steps)
  global_step = get_global_step(hparams)

  if warmup_schedule == "exp":
    return tf.exp(tf.log(0.01) / warmup_steps)**(warmup_steps - global_step)
  else:
    assert warmup_schedule == "linear"
    start = tf.constant(0.35)
    return ((tf.constant(1.) - start) / warmup_steps) * global_step + start


def learning_rate_decay_with_warmup(hparams, num_worker_replicas=1):
  """Learning rate decay rate with warmup based on hparams."""
  warmup_steps = hparams.learning_rate_warmup_steps * num_worker_replicas
  warmup = learning_rate_warmup(warmup_steps, hparams=hparams)
  
  decay = learning_rate_decay(hparams, warmup_steps)
  global_step = get_global_step(hparams)
  return tf.where(global_step < warmup_steps, warmup, decay)


def weight_decay_and_noise(loss, hparams, learning_rate, var_list=None):
  """Apply weight decay and weight noise."""
  if var_list is None:
    var_list = tf.trainable_variables()

  decay_vars = [v for v in var_list]
  noise_vars = [v for v in var_list if "/body/" in v.name]

  weight_decay_loss = weight_decay(hparams.weight_decay, decay_vars)
  tf.summary.scalar("losses/weight_decay", weight_decay_loss)
  weight_noise_ops = weight_noise(hparams.weight_noise, learning_rate,
                                  noise_vars)

  with tf.control_dependencies(weight_noise_ops):
    loss = tf.identity(loss)

  loss += weight_decay_loss
  return loss


def weight_noise(noise_rate, learning_rate, var_list):
  """Apply weight noise to vars in var_list."""
  if not noise_rate:
    return [tf.no_op()]

  tf.logging.info("Applying weight noise scaled by learning rate, "
                  "noise_rate: %0.5f", noise_rate)

  noise_ops = []

  for v in var_list:
    with tf.device(v._ref().device):  # pylint: disable=protected-access
      scale = noise_rate * learning_rate * 0.001
      tf.summary.scalar("weight_noise_scale", scale)
      noise = tf.truncated_normal(v.shape) * scale
      noise_op = v.assign_add(noise)
      noise_ops.append(noise_op)

  return noise_ops


def weight_decay(decay_rate, var_list, skip_biases=True):
  """Apply weight decay to vars in var_list."""
  if not decay_rate:
    return 0.

  tf.logging.info("Applying weight decay, decay_rate: %0.5f", decay_rate)

  weight_decays = []
  for v in var_list:
    # Weight decay.
    # This is a heuristic way to detect biases that works for main tf.layers.
    is_bias = len(v.shape.as_list()) == 1 and v.name.endswith("bias:0")
    if not (skip_biases and is_bias):
      with tf.device(v.device):
        v_loss = tf.nn.l2_loss(v)
      weight_decays.append(v_loss)

  return tf.add_n(weight_decays) * decay_rate


def log_variable_sizes(var_list=None, tag=None, verbose=False):
  """Log the sizes and shapes of variables, and the total size.

  Args:
    var_list: a list of variables; defaults to trainable_variables
    tag: a string; defaults to "Trainable Variables"
    verbose: bool, if True, log every weight; otherwise, log total size only.
  """
  if var_list is None:
    var_list = tf.trainable_variables()
  if tag is None:
    tag = "Trainable Variables"

  if not var_list:
    return

  name_to_var = {v.name: v for v in var_list}
  total_size = 0
  for v_name in sorted(list(name_to_var)):
    v = name_to_var[v_name]
    v_size = int(np.prod(np.array(v.shape.as_list())))
    if verbose:
      tf.logging.info("Weight    %s\tshape    %s\tsize    %d",
                      v.name[:-2].ljust(80),
                      str(v.shape).ljust(20), v_size)
    total_size += v_size
  tf.logging.info("%s Total size: %d", tag, total_size)


def get_variable_initializer(hparams):
  """Get variable initializer from hparams."""
  if not hparams.initializer:
    return None

  tf.logging.info("Using variable initializer: %s", hparams.initializer)
  if hparams.initializer == "orthogonal":
    return tf.orthogonal_initializer(gain=hparams.initializer_gain)
  elif hparams.initializer == "uniform":
    max_val = 0.1 * hparams.initializer_gain
    return tf.random_uniform_initializer(-max_val, max_val)
  elif hparams.initializer == "normal_unit_scaling":
    return tf.variance_scaling_initializer(
        hparams.initializer_gain, mode="fan_avg", distribution="normal")
  elif hparams.initializer == "uniform_unit_scaling":
    return tf.variance_scaling_initializer(
        hparams.initializer_gain, mode="fan_avg", distribution="uniform")
  else:
    raise ValueError("Unrecognized initializer: %s" % hparams.initializer)


