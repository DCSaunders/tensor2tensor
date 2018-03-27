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

"""Summary-based SessionRunHooks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

import tensorflow as tf

TMP_NAME = "dev.out-tmp"

class EvalFileOutHook(tf.train.SessionRunHook):
  """Hook to handle validation output tracking
  """
  _EVAL_NAME = "dev.out-%d"


  def __init__(self, eval_file_out_dir, save_last_only=True):
    """Construct EvalFileOutHook.

    Args:
      eval_file_out_dir: str, directory containing eval out files
      save_every_eval: bool, true if keeping eval out file for every checkpoint
    """
    self.eval_file_out_dir = eval_file_out_dir
    self.save_last_only = save_last_only
    self._start_step = None
    self.global_step = None
    self.prev_checkpoint_step = None
    self.remove_prev = False

  def begin(self):
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError("Global step must be created to track eval outputs.")

  def after_create_session(self, session, coord):
    del coord
    if self._start_step is None:
      self._start_step = session.run(self._global_step_tensor)

  def before_run(self, run_context):
    del run_context
    return tf.train.SessionRunArgs([self._global_step_tensor])

  def after_run(self, run_context, run_values):
    if self.global_step != run_values.results[0]:
      self.prev_checkpoint_step = self.global_step
      self.global_step = run_values.results[0]
      self.remove_prev = True

  def end(self, session):
    out_file = self._EVAL_NAME % self.global_step
    tmp_path = os.path.join(self.eval_file_out_dir, TMP_NAME)
    out_path = os.path.join(self.eval_file_out_dir, out_file)
    os.rename(tmp_path, out_path)
    if self.save_last_only and self.remove_prev:
      self.remove_previous()

  def remove_previous(self):
    if self.prev_checkpoint_step is not None and self.prev_checkpoint_step != self.global_step:
      prev_file = self._EVAL_NAME % self.prev_checkpoint_step
      prev_path = os.path.join(self.eval_file_out_dir, prev_file)
      try:
        os.remove(prev_path)
        self.remove_prev = False
      except OSError:
        tf.logging.warning('Unable to remove previous eval out: {}'.format(prev_path))