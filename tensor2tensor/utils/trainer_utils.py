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

"""Utilities for trainer binary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import operator
import os
import re
import sys

# Dependency imports

from tensor2tensor import models  # pylint: disable=unused-import
from tensor2tensor.data_generators import all_problems  # pylint: disable=unused-import
from tensor2tensor.utils import data_reader
from tensor2tensor.utils import decoding
from tensor2tensor.utils import devices
from tensor2tensor.utils import input_fn_builder
from tensor2tensor.utils import model_builder
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics

import tensorflow as tf
from tensorflow.contrib.hooks.python.training.profiler_hook import ProfilerHook
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python import debug
from tensorflow.contrib.learn.python.learn import monitors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_lib

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool("registry_help", False,
                  "If True, logs the contents of the registry and exits.")
flags.DEFINE_bool("tfdbg", False,
                  "If True, use the TF debugger CLI on train/eval.")
flags.DEFINE_bool("export_saved_model", False,
                  "Whether to export a SavedModel for serving.")
flags.DEFINE_bool("dbgprofile", False,
                  "If True, record the timeline for chrome://tracing/.")
flags.DEFINE_string("model", "", "Which model to use.")
flags.DEFINE_string("hparams_set", "", "Which parameters to use.")
flags.DEFINE_string("hparams_range", "", "Parameters range.")
flags.DEFINE_string(
    "hparams", "",
    """A comma-separated list of `name=value` hyperparameter values. This flag
    is used to override hyperparameter settings either when manually selecting
    hyperparameters or when using Vizier. If a hyperparameter setting is
    specified by this flag then it must be a valid hyperparameter name for the
    model.""")
flags.DEFINE_string("problems", "", "Dash separated list of problems to "
                    "solve.")
flags.DEFINE_string("data_dir", "/tmp/data", "Directory with training data.")
flags.DEFINE_integer("train_steps", 250000,
                     "The number of steps to run training for.")
flags.DEFINE_string("target_metric", "approx_bleu_score", "Metric which is "
                    "used to select the models to store in train_dir/best: "
                    "accuracy_per_sequence, accuracy_top5, accuracy, "
                    "neg_log_perplexity, approx_bleu_score.")
flags.DEFINE_bool("eval_run_autoregressive", False,
                  "Run eval autoregressively where we condition on previous"
                  "generated output instead of the actual target.")
flags.DEFINE_integer("keep_checkpoint_max", 10,
                     "How many recent checkpoints to keep.")
flags.DEFINE_bool("experimental_optimize_placement", False,
                  "Optimize ops placement with experimental session options.")
flags.DEFINE_integer("keep_checkpoint_every_n_hours", 10000,
                     "Number of hours between each checkpoint to be saved. "
                     "The default value 10,000 hours effectively disables it.")
flags.DEFINE_integer("save_checkpoints_secs", 0,
                     "Save checkpoints every this many seconds. "
                     "Default=0 means let tensorflow.contrib.learn.python.learn"
                     " decide, which is currently set to 600 = 10 minutes.")
flags.DEFINE_bool("log_device_placement", False,
                  "Whether to log device placement.")

# Distributed training flags
flags.DEFINE_integer("local_eval_frequency", 2000,
                     "Run evaluation every this steps during local training.")
flags.DEFINE_bool("locally_shard_to_cpu", False,
                  "Use CPU as a sharding device running locally. This allows "
                  "to test sharded model construction on a machine with 1 GPU.")
flags.DEFINE_bool("daisy_chain_variables", True,
                  "copy variables around in a daisy chain")
flags.DEFINE_bool("sync", False, "Sync compute on PS.")
flags.DEFINE_string("worker_job", "/job:localhost", "name of worker job")
flags.DEFINE_integer("worker_gpu", 1, "How many GPUs to use.")
flags.DEFINE_integer("worker_replicas", 1, "How many workers to use.")
flags.DEFINE_integer("worker_id", 0, "Which worker task are we.")
flags.DEFINE_float("worker_gpu_memory_fraction", 0.95,
                   "Fraction of GPU memory to allocate.")
flags.DEFINE_integer("ps_gpu", 0, "How many GPUs to use per ps.")
flags.DEFINE_string("gpu_order", "", "Optional order for daisy-chaining gpus."
                    " e.g. \"1 3 2 4\"")
flags.DEFINE_string("ps_job", "/job:ps", "name of ps job")
flags.DEFINE_integer("ps_replicas", 0, "How many ps replicas.")

# Decoding flags
flags.DEFINE_string(
    "decode_hparams", "",
    "Comma-separated list of name=value pairs to control decode behavior. "
    "See decoding.decode_hparams for defaults.")


class SaveBestCheckpointsMonitor(monitors.ValidationMonitor):

  def _scan_best_dir(self, best_dir):
    existing_checkpoints = []
    for file_name in tf.gfile.Glob("%s/model.ckpt-*.index" % best_dir):
      match = re.search(r"score([.0-9]+).index$", file_name)
      if match:
        existing_checkpoints.append((float(match.group(1)), file_name[:-6]))
      else:
        logging.info("Checkpoint %s did not match pattern." % file_name)
    return existing_checkpoints

  def _copy_checkpoint(self, src, trg):
    logging.info("Create new best checkpoint file: %s" % trg)
    for src_file in tf.gfile.Glob("%s.*" % src):
      ext = os.path.splitext(src_file)[1]
      tf.gfile.Copy(src_file, "%s%s" % (trg, ext))

  def _remove_checkpoint(self, path):
    logging.info("Remove from best checkpoint dir: %s" % path)
    for file_path in tf.gfile.Glob("%s.*" % path):
      tf.gfile.Remove(file_path)

  def _rebuild_checkpoint_file(self, path, checkpoints):
    logging.info("Rebuild checkpoint file with %d entries: %s"
      % (len(checkpoints), path))
    prefix = "%s/" % os.path.dirname(path)
    l = len(prefix)
    paths = [p[l:] if p.startswith(prefix) else p for p in checkpoints]
    with tf.gfile.Open(path, "w") as f:
      f.write('model_checkpoint_path: "%s"\n' % paths[-1])
      for checkpoint_path in paths:
        f.write('all_model_checkpoint_paths: "%s"\n' % checkpoint_path)

  def every_n_step_end(self, step, outputs):
    # TODO(mdan): The use of step below is probably misleading.
    # The code should probably use the step from the checkpoint, because
    # that's what is being evaluated.
    if self._estimator is None:
      raise ValueError("Missing call to set_estimator.")
    # Check that we are not running evaluation on the same checkpoint.
    latest_path = saver_lib.latest_checkpoint(self._estimator.model_dir)
    if latest_path is None:
      logging.debug("Skipping evaluation since model has not been saved yet "
                    "at step %d.", step)
      return False
    if latest_path is not None and latest_path == self._latest_path:
      logging.debug("Skipping evaluation due to same checkpoint %s for step %d "
                    "as for step %d.", latest_path, step,
                    self._latest_path_step)
      return False
    self._latest_path = latest_path
    self._latest_path_step = step

    # Run evaluation and log it.
    validation_outputs = self._evaluate_estimator()
    stats = []
    for name in validation_outputs:
      stats.append("%s = %s" % (name, str(validation_outputs[name])))
    logging.info("Validation (step %d): %s", step, ", ".join(stats))

    # Store best checkpoints logic
    best_dir = "%s/best" % os.path.dirname(latest_path)
    tf.gfile.MakeDirs(best_dir)
    current_best_checkpoints = self._scan_best_dir(best_dir)
    current_best_checkpoints.append(
      (validation_outputs[self.early_stopping_metric], None))
    current_best_checkpoints.sort(key=operator.itemgetter(0),
                                  reverse=self.early_stopping_metric_minimize)
    new_best_paths = []
    rebuild_checkpoint_file = False
    for score, path in current_best_checkpoints[-self.early_stopping_rounds:]:
      if path is None:  # This is the new checkpoint
        rebuild_checkpoint_file = True
        path = "%s/model.ckpt-%d_score%.4f" % (best_dir, step, score)
        self._copy_checkpoint(latest_path, path)
      new_best_paths.append(path)
    for _, path in current_best_checkpoints[:-self.early_stopping_rounds]:
      if path is not None:
        rebuild_checkpoint_file = True
        self._remove_checkpoint(path)
    if rebuild_checkpoint_file:
      self._rebuild_checkpoint_file("%s/checkpoint" % best_dir,
                                    new_best_paths)


def make_experiment_fn(data_dir, model_name, train_steps, eval_steps):
  """Returns experiment_fn for learn_runner. Wraps create_experiment."""

  def experiment_fn(run_config, hparams):
    return create_experiment(
        data_dir,
        model_name=model_name,
        train_steps=train_steps,
        eval_steps=eval_steps,
        hparams=hparams,
        run_config=run_config)

  return experiment_fn


def create_experiment(data_dir, model_name, train_steps, eval_steps, hparams,
                      run_config):
  """Create Experiment."""
  estimator, input_fns = create_experiment_components(
      data_dir=data_dir,
      model_name=model_name,
      hparams=hparams,
      run_config=run_config)

  train_monitors = []
  eval_hooks = []
  if FLAGS.tfdbg:
    hook = debug.LocalCLIDebugHook()
    train_monitors.append(hook)
    eval_hooks.append(hook)
  if FLAGS.dbgprofile:
    # Recorded traces can be visualized with chrome://tracing/
    # The memory/tensor lifetime is also profiled
    train_monitors.append(ProfilerHook(
        save_steps=10,
        output_dir=run_config.model_dir,
        show_dataflow=True,
        show_memory=True,
    ))

  optional_kwargs = {}
  if FLAGS.export_saved_model:
    assert len(hparams.problem_instances) == 1
    problem = hparams.problem_instances[0]
    optional_kwargs["export_strategies"] = [
        make_export_strategy(problem, hparams)
    ]
  if FLAGS.local_eval_frequency:
    eval_metrics = metrics.create_evaluation_metrics(
      zip(FLAGS.problems.split("-"), hparams.problem_instances), hparams)
    stopping_metric_name = None
    for metric_name in eval_metrics:
      if metric_name.endswith("/%s" % FLAGS.target_metric):
        stopping_metric_name = metric_name
        break
    if stopping_metric_name is None:
      raise ValueError("--target_metric did not match any known metric.")
    logging.info("Selected target metric: %s" % stopping_metric_name)
    train_monitors.append(SaveBestCheckpointsMonitor(
      input_fn=input_fns["eval"], eval_steps=eval_steps,
      every_n_steps=FLAGS.local_eval_frequency,
      name=None, hooks=eval_hooks,
      early_stopping_metric=stopping_metric_name,
      early_stopping_metric_minimize=False,
      early_stopping_rounds=FLAGS.keep_checkpoint_max
    ))

  return tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=input_fns[tf.estimator.ModeKeys.TRAIN],
      eval_input_fn=input_fns[tf.estimator.ModeKeys.EVAL],
      train_steps=train_steps,
      eval_steps=eval_steps,
      min_eval_frequency=0,  # Validation is done in SaveBestCheckpointsMonitor
      train_monitors=train_monitors,
      eval_hooks=eval_hooks,
      **optional_kwargs)


def make_export_strategy(problem, hparams):
  return tf.contrib.learn.make_export_strategy(
      lambda: data_reader.serving_input_fn(problem, hparams), as_text=True)


def create_experiment_components(data_dir, model_name, hparams, run_config):
  """Constructs and returns Estimator and train/eval input functions."""
  tf.logging.info("Creating experiment, storing model files in %s",
                  run_config.model_dir)

  hparams = add_problem_hparams(hparams, FLAGS.problems)

  num_datashards = devices.data_parallelism().n
  train_input_fn = input_fn_builder.build_input_fn(
      mode=tf.estimator.ModeKeys.TRAIN,
      hparams=hparams,
      data_file_patterns=get_data_filepatterns(data_dir,
                                               tf.estimator.ModeKeys.TRAIN),
      num_datashards=num_datashards,
      worker_replicas=FLAGS.worker_replicas,
      worker_id=FLAGS.worker_id)

  eval_input_fn = input_fn_builder.build_input_fn(
      mode=tf.estimator.ModeKeys.EVAL,
      hparams=hparams,
      data_file_patterns=get_data_filepatterns(data_dir,
                                               tf.estimator.ModeKeys.EVAL),
      num_datashards=num_datashards,
      worker_replicas=FLAGS.worker_replicas,
      worker_id=FLAGS.worker_id)

  model_fn = model_builder.build_model_fn(
      model_name,
      problem_names=FLAGS.problems.split("-"),
      train_steps=FLAGS.train_steps,
      worker_id=FLAGS.worker_id,
      worker_replicas=FLAGS.worker_replicas,
      eval_run_autoregressive=FLAGS.eval_run_autoregressive,
      decode_hparams=decoding.decode_hparams(FLAGS.decode_hparams))

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=run_config.model_dir,
      params=hparams,
      config=run_config)

  return estimator, {
      tf.estimator.ModeKeys.TRAIN: train_input_fn,
      tf.estimator.ModeKeys.EVAL: eval_input_fn
  }


def log_registry():
  if FLAGS.registry_help:
    tf.logging.info(registry.help_string())
    sys.exit(0)


def add_problem_hparams(hparams, problems):
  """Add problem hparams for the problems."""
  hparams.problems = []
  hparams.problem_instances = []
  for problem_name in problems.split("-"):
    try:
      problem = registry.problem(problem_name)
    except LookupError:
      all_problem_names = sorted(registry.list_problems())
      error_lines = ["%s not in the set of supported problems:" % problem_name
                    ] + all_problem_names
      error_msg = "\n  * ".join(error_lines)
      raise LookupError(error_msg)
    p_hparams = problem.get_hparams(hparams)

    hparams.problem_instances.append(problem)
    hparams.problems.append(p_hparams)

  return hparams


def save_metadata(output_dir, hparams):
  """Saves FLAGS and hparams to output_dir."""
  # Save FLAGS in txt file
  if hasattr(FLAGS, "flags_into_string"):
    flags_str = FLAGS.flags_into_string()
    t2t_flags_str = "\n".join([
        "--%s=%s" % (f.name, f.value)
        for f in FLAGS.flags_by_module_dict()[
            "tensor2tensor.utils.trainer_utils"]
    ])
  else:
    flags_dict = FLAGS.__dict__["__flags"]
    flags_str = "\n".join(
        ["--%s=%s" % (name, str(f)) for (name, f) in flags_dict.items()])
    t2t_flags_str = None

  flags_txt = os.path.join(output_dir, "flags.txt")
  with tf.gfile.Open(flags_txt, "w") as f:
    f.write(flags_str)

  if t2t_flags_str:
    t2t_flags_txt = os.path.join(output_dir, "flags_t2t.txt")
    with tf.gfile.Open(t2t_flags_txt, "w") as f:
      f.write(t2t_flags_str)

  # Save hparams as hparams.json
  hparams_fname = os.path.join(output_dir, "hparams.json")
  with tf.gfile.Open(hparams_fname, "w") as f:
    f.write(hparams.to_json())


def create_hparams(params_id, data_dir, passed_hparams=None):
  """Returns hyperparameters, including any flag value overrides.

  If the hparams FLAG is set, then it will use any values specified in
  hparams to override any individually-set hyperparameter. This logic
  allows tuners to override hyperparameter settings to find optimal values.

  Args:
    params_id: which set of parameters to choose (must be in _PARAMS above).
    data_dir: the directory containing the training data.
    passed_hparams: command-line overrides for some hparams.

  Returns:
    The hyperparameters as a tf.contrib.training.HParams object.
  """
  hparams = registry.hparams(params_id)()
  hparams.add_hparam("data_dir", data_dir)
  # Command line flags override any of the preceding hyperparameter values.
  if passed_hparams:
    hparams = hparams.parse(passed_hparams)

  return hparams


def create_run_config(output_dir):
  """Create a RunConfig object."""

  run_config = tf.contrib.learn.RunConfig(
      model_dir=output_dir,
      master=FLAGS.master,
      gpu_memory_fraction=FLAGS.worker_gpu_memory_fraction,
      session_config=session_config(),
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs)

  return run_config


def run(data_dir, model, output_dir, train_steps, eval_steps, schedule):
  """Runs an Estimator locally or distributed.

  Args:
    data_dir: The directory the data can be found in.
    model: The name of the model to use.
    output_dir: The directory to store outputs in.
    train_steps: The number of steps to run training for.
    eval_steps: The number of steps to run evaluation for.
    schedule: (str) The schedule to run. The value here must
      be the name of one of Experiment's methods.
  """
  exp_fn = make_experiment_fn(
      data_dir=data_dir,
      model_name=model,
      train_steps=train_steps,
      eval_steps=eval_steps)

  # Create hparams and run_config
  run_config = create_run_config(output_dir)
  hparams = create_hparams(
      FLAGS.hparams_set, data_dir, passed_hparams=FLAGS.hparams)

  if is_chief():
    save_metadata(output_dir, hparams)

  learn_runner.run(
      experiment_fn=exp_fn,
      schedule=schedule,
      run_config=run_config,
      hparams=hparams)


def validate_flags():
  if not FLAGS.model:
    raise ValueError("Must specify a model with --model.")
  if not FLAGS.problems:
    raise ValueError("Must specify a set of problems with --problems.")
  if not (FLAGS.hparams_set or FLAGS.hparams_range):
    raise ValueError("Must specify either --hparams_set or --hparams_range.")
  if not FLAGS.schedule:
    raise ValueError("Must specify --schedule.")
  if not FLAGS.output_dir:
    FLAGS.output_dir = "/tmp/tensor2tensor"
    tf.logging.warning("It is strongly recommended to specify --output_dir. "
                       "Using default output_dir=%s.", FLAGS.output_dir)


def is_chief():
  schedules = ["train", "train_and_evaluate"]
  return FLAGS.worker_id == 0 and FLAGS.schedule in schedules


def session_config():
  """The TensorFlow Session config to use."""
  graph_options = tf.GraphOptions(optimizer_options=tf.OptimizerOptions(
      opt_level=tf.OptimizerOptions.L1, do_function_inlining=False))

  if FLAGS.experimental_optimize_placement:
    rewrite_options = tf.RewriterConfig(optimize_tensor_layout=True)
    rewrite_options.optimizers.append("pruning")
    rewrite_options.optimizers.append("constfold")
    rewrite_options.optimizers.append("layout")
    graph_options = tf.GraphOptions(
        rewrite_options=rewrite_options, infer_shapes=True)

  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=FLAGS.worker_gpu_memory_fraction)

  config = tf.ConfigProto(
      allow_soft_placement=True,
      graph_options=graph_options,
      gpu_options=gpu_options,
      log_device_placement=FLAGS.log_device_placement)
  return config


def get_data_filepatterns(data_dir, mode):
  return data_reader.get_data_filepatterns(FLAGS.problems, data_dir, mode)
