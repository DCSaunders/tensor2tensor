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
import os
import sys
import cPickle as pickle
from tensor2tensor.utils import yellowfin
from tensor2tensor.utils import largebatch_optimizer
import tensorflow as tf
from tensorflow.python.util import nest

from tensorflow.python.framework import dtypes
from tensorflow.python.eager import context

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope

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



class EWCOptimizer(ConditionalOptimizer):
  def __init__(self, optimizer_name, lr, hparams, use_tpu=False):
    super(EWCOptimizer, self).__init__(optimizer_name, lr, hparams, use_tpu)
    self.load_vars = hparams.ewc_load_vars
    self.ignore_fisher = hparams.ewc_ignore_fisher
    self.save_vars = hparams.ewc_save_vars
    self.model_dir = hparams.model_dir
    self.get_checkpoints_and_weights(hparams)
    self.lag_vals_to_save = []
    self.fisher_vals_to_save = []
    self.lag_vals = []
    self.fisher_vals = []

    self.final_step = hparams.train_steps
    self.save_step = 0
    self.fisher_accum_steps = 1
    self.first_save_step = -1
    self.set_steps(hparams)
    if self.load_vars:
      self.load_ewc_vals()
      
  def get_checkpoints_and_weights(self, hparams):
    self.ewc_checkpoints = [os.path.join(self.model_dir, ckpt) 
          for ckpt in hparams.ewc_checkpoint.split(';')]
    self.ewc_checkpoint_to_save = os.path.join(self.model_dir, 
                                               hparams.ewc_checkpoint_to_save)
    try:
      self.loss_weights = [float(weight) 
          for weight in hparams.ewc_loss_weight.split(';')]
    except AttributeError:
      self.loss_weights = [float(hparams.ewc_loss_weight)]

  def load_ewc_vals(self):
    for ckpt in self.ewc_checkpoints:
      ckpt_fisher = []
      ckpt_lagged = []
      with open(ckpt, 'rb') as fp:
        tf.logging.info('Loading lagged vals from {}'.format(ckpt))
        vals = pickle.load(fp)
        for fisher, lag in vals:
          ckpt_fisher.append(fisher)
          ckpt_lagged.append(lag)
      self.fisher_vals.append(ckpt_fisher)
      self.lag_vals.append(ckpt_lagged)


  def set_steps(self, hparams):
    if self.save_vars:
      self.save_step = hparams.train_steps - hparams.ewc_fisher_accum_steps
      tf.logging.info('Train {} steps before accumulating fisher'.format(self.save_step))
      self.fisher_accum_steps = hparams.ewc_fisher_accum_steps
      tf.logging.info('Then accumulate for {} steps'.format(self.fisher_accum_steps))
      FLAGS = tf.flags.FLAGS
      if self.fisher_accum_steps > FLAGS.local_eval_frequency:
        tf.logging.warning('Fisher accumulation is reset during evaluation')
        tf.logging.warning(
          'To accumulate for > {} steps, increase local_eval_frequency'.format(
            FLAGS.local_eval_frequency))


  def update_ewc_vals(self, *grads_vars_and_step):
    global_step = grads_vars_and_step[-1]
    if self.first_save_step < 0:
      self.first_save_step = self.save_step
    last_step = (global_step >= self.first_save_step + self.fisher_accum_steps - 1)
    tf.logging.debug('Updating EWC vars: step {}'.format(global_step))
    for idx, grad_var_pair in enumerate(grads_vars_and_step[:-1]):
      fisher_val = np.square(grad_var_pair[0]) / self.fisher_accum_steps
      if idx == len(self.fisher_vals_to_save):
        self.fisher_vals_to_save.append(fisher_val)
      else:
        self.fisher_vals_to_save[idx] += fisher_val
      if last_step:
        self.lag_vals_to_save.append(grad_var_pair[1])
    if last_step:
      tf.logging.info('Last step of accumulation: pickling EWC variables')
      with open(self.ewc_checkpoint_to_save, 'wb') as fp:
        pickle.dump(zip(self.fisher_vals_to_save, self.lag_vals_to_save), fp)
      sys.exit('Saved EWC vars, exiting')
    return 1
    
  def accumulate_ewc(self, grads_and_vars, global_step, name):
    updates_to_ignore = self._opt.apply_gradients(grads_and_vars, global_step, name)
    
    step = tf.py_func(self.update_ewc_vals, grads_and_vars + [global_step], tf.int64)
    ewc_ops = [global_step.assign_add(step)]
    return control_flow_ops.group(*ewc_ops)
    

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    v_list = [v for (_, v) in grads_and_vars]
    self._opt._create_slots(v_list)
    fisher_cond = tf.logical_and(tf.constant(self.save_vars, dtype=tf.bool),
                                 tf.greater_equal(global_step, self.save_step))
    maybe_accumulate_fisher = tf.cond(fisher_cond,
      lambda: self.accumulate_ewc(grads_and_vars, global_step, name=name),
      lambda: self._opt.apply_gradients(grads_and_vars, global_step, name=name),
      name=name)
    return maybe_accumulate_fisher

  def hacky_print(self, t):
    tf.logging.info(t)
    return np.float32(0.0)

  def lagged_loss(self, fisher, lagged, weight):
    if self.ignore_fisher:
      ewc_losses = [tf.reduce_sum(tf.square(l - t))
          for l, t in zip(lagged, tf.trainable_variables())]
    else:
      ewc_losses = [tf.reduce_sum(tf.square(l - t) * f)
          for l, t, f in zip(lagged, tf.trainable_variables(), fisher)]
    return weight * tf.add_n(ewc_losses)


  def get_ewc_loss(self):
    tf.logging.info('Adding EWC penalty to loss with lambda(s) {}'.format(self.loss_weights))
    ewc_loss = 0
    for f, l, w in zip(self.fisher_vals, self.lag_vals, self.loss_weights):
      ewc_loss += self.lagged_loss(f, l, w)
    return ewc_loss
