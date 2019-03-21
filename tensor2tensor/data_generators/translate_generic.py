"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import lm1b
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS


# End-of-sentence marker.
EOS = text_encoder.EOS_ID


# Data-set location

_TRAIN_DATASETS = [("train/training.src",
                    "train/training.trg")]

_TEST_DATASETS = [("dev/dev.src",
                   "dev/dev.trg")]

_VOCABS = ("vocab/vocab.src",
           "vocab/vocab.trg",
           "vocab/vocab.shared")

_SCALE = ("train/scale")

def _compile_data(tmp_dir, datasets, filename):
  """Concatenate all `datasets` and save to `filename`."""
  filename = os.path.join(tmp_dir, filename)
  with tf.gfile.GFile(filename + ".src", mode="w") as src_resfile:
    with tf.gfile.GFile(filename + ".trg", mode="w") as trg_resfile:
      for dataset in datasets:
        orig_dir = FLAGS.raw_data_dir
        src_filename, trg_filename = dataset
        src_filepath = os.path.join(orig_dir, src_filename)
        trg_filepath = os.path.join(orig_dir, trg_filename)
        is_sgm = (src_filename.endswith("sgm") and
                  trg_filename.endswith("sgm"))

        if not (os.path.exists(src_filepath) and
                os.path.exists(trg_filepath)):
            raise ValueError("Can't find files %s %s." % (src_filepath, trg_filepath))
        with tf.gfile.GFile(src_filepath, mode="r") as src_file:
          with tf.gfile.GFile(trg_filepath, mode="r") as trg_file:
            line1, line2 = src_file.readline(), trg_file.readline()
            while line1 or line2:
              src_resfile.write(line1.strip() + "\n")
              trg_resfile.write(line2.strip() + "\n")
              line1, line2 = src_file.readline(), trg_file.readline()

  return filename


def _compile_monolingual_data(tmp_dir, datasets, filename):
  """Concatenate all `datasets` and save to `filename`."""
  filename = os.path.join(tmp_dir, filename)
  with tf.gfile.GFile(filename + ".trg", mode="w") as trg_resfile:
    for dataset in datasets:
      orig_dir = FLAGS.raw_data_dir
      trg_filename = dataset[1]
      trg_filepath = os.path.join(orig_dir, trg_filename)
      is_sgm = (trg_filename.endswith("sgm"))
      if not os.path.exists(trg_filepath):
        raise ValueError("Can't find files %s %s." % (trg_filepath))
      with tf.gfile.GFile(trg_filepath, mode="r") as trg_file:
        line = trg_file.readline()
        while line:
          trg_resfile.write(line.strip() + "\n")
          line = trg_file.readline()
  return filename


def copyVocab(orig_path, target_path):
    # Add <pad> (idx:0), <EOS> (idx:1) and <unk> (idx:2) to vocab
    with tf.gfile.GFile(target_path, mode="w") as f:
        f.write("<pad>\n<EOS>\n<unk>\n");
        with tf.gfile.Open(orig_path) as origF:
            for line in origF:
                tokens = line.strip().split("\t")
                # To handle 3-column vocab file: "<idx>\t<token>\t<count>"
                if len(tokens) > 1:
                    f.write("%s\n" % tokens[1])
                else:
                    f.write("%s\n" % tokens[0])

@registry.register_problem
class LanguagemodelGenericExistingVocab(lm1b.LanguagemodelLm1b32k):

  @property
  def target_vocab_name(self):
    return "vocab.trg"

  @property
  def has_inputs(self):
    return False

  @property
  def targeted_vocab_size(self):
    return 32126

  def generator(self, data_dir, tmp_dir, is_training):
    """Generator for lm1b sentences.

    Args:
      data_dir: data dir.
      tmp_dir: tmp dir.
      is_training: a boolean.

    Yields:
      A dictionary {"inputs": [0], "targets": [<subword ids>]}
    """
    possible_datasets = _TRAIN_DATASETS if is_training else _TEST_DATASETS
    datasets = [[FLAGS.raw_data_dir, [item[1]]] for item in possible_datasets]
    target_vocab_path = os.path.join(data_dir, self.target_vocab_name)

    copyVocab(os.path.join(FLAGS.raw_data_dir, _VOCABS[1]), target_vocab_path)
    token_vocab = text_encoder.TokenTextEncoder(target_vocab_path, replace_oov="<unk>")
    tag = "train" if is_training else "dev"
    data_path = _compile_monolingual_data(tmp_dir, possible_datasets, "generic_tok_%s" % tag)
    tf.logging.info("filepath = %s", data_path)
    with tf.gfile.GFile(data_path + '.trg', mode="r") as in_file:
      line = in_file.readline()
      while line:
        tokens = token_vocab.encode(line.strip()) 
        tokens.append(EOS)
        yield {"targets": tokens}
        line = in_file.readline()

  def feature_encoders(self, data_dir):
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    target_encoder = text_encoder.TokenTextEncoder(target_vocab_filename, replace_oov="<unk>")
    return {"targets": target_encoder}



@registry.register_problem
class TranslateGeneric(translate.TranslateProblem):
  """Problem spec for generic wordpiece translation."""

  @property
  def input_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def target_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def targeted_vocab_size(self):
    return FLAGS.targeted_vocab_size

  @property
  def source_vocab_name(self):
    return "vocab.src.%d" % self.targeted_vocab_size

  @property
  def target_vocab_name(self):
    return "vocab.trg.%d" % self.targeted_vocab_size

  def generator(self, data_dir, tmp_dir, train):
    datasets = _TRAIN_DATASETS if train else _TEST_DATASETS
    source_datasets = [[FLAGS.raw_data_dir, [item[0]]] for item in datasets]
    target_datasets = [[FLAGS.raw_data_dir, [item[1]]] for item in datasets]
    source_vocab = generator_utils.get_or_generate_vocab_nocompress(
        data_dir, self.source_vocab_name,
        self.targeted_vocab_size, source_datasets)
    target_vocab = generator_utils.get_or_generate_vocab_nocompress(
        data_dir, self.target_vocab_name,
        self.targeted_vocab_size, target_datasets)
    tag = "train" if train else "dev"
    data_path = _compile_data(tmp_dir, datasets, "generic_tok_%s" % tag)
    return translate.bi_vocabs_token_generator(data_path + ".src", data_path + ".trg",
                                     source_vocab, target_vocab, EOS)

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir,
                                         self.source_vocab_name)
    target_vocab_filename = os.path.join(data_dir,
                                         self.target_vocab_name)
    source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
    target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
    return {
        "inputs": source_token,
        "targets": target_token,
    }


@registry.register_problem
class TranslateGenericExistingVocab(translate.TranslateProblem):
  """Problem spec for generic translation, using existing vocab """

  @property
  def input_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def target_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def source_vocab_name(self):
    return "vocab.src"

  @property
  def target_vocab_name(self):
    return "vocab.trg"

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    source_encoder = text_encoder.TokenTextEncoder(source_vocab_filename, replace_oov="<unk>")
    target_encoder = text_encoder.TokenTextEncoder(target_vocab_filename, replace_oov="<unk>")
    return {"inputs": source_encoder, "targets": target_encoder}

  def generator(self, data_dir, tmp_dir, train):
    datasets = _TRAIN_DATASETS if train else _TEST_DATASETS
    source_datasets = [[FLAGS.raw_data_dir, [item[0]]] for item in datasets]
    target_datasets = [[FLAGS.raw_data_dir, [item[1]]] for item in datasets]
    # Copy vocab to data directory
    source_vocab_path = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_path = os.path.join(data_dir, self.target_vocab_name)
    if os.path.exists(source_vocab_path):
        os.remove(source_vocab_path)
    if os.path.exists(target_vocab_path):
        os.remove(target_vocab_path)
    copyVocab(os.path.join(FLAGS.raw_data_dir, _VOCABS[0]), source_vocab_path)
    copyVocab(os.path.join(FLAGS.raw_data_dir, _VOCABS[1]), target_vocab_path)
    source_token_vocab = text_encoder.TokenTextEncoder(source_vocab_path, replace_oov="<unk>")
    target_token_vocab = text_encoder.TokenTextEncoder(target_vocab_path, replace_oov="<unk>")
    tag = "train" if train else "dev"
    data_path = _compile_data(tmp_dir, datasets, "generic_tok_%s" % tag)
    return translate.bi_vocabs_token_generator(data_path + ".src", data_path + ".trg",
                                     source_token_vocab, target_token_vocab, EOS)


@registry.register_problem
class TranslateGenericExistingSharedVocab(translate.TranslateProblem):
  """Problem spec for generic translation, using existing vocab
  which is shared between source and target """

  @property
  def input_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def target_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def vocab_name(self):
    return "vocab.shared"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_name)
    encoder = text_encoder.TokenTextEncoder(vocab_filename, replace_oov="<unk>")
    return {"inputs": encoder, "targets": encoder}

  def generator(self, data_dir, tmp_dir, train):
    datasets = _TRAIN_DATASETS if train else _TEST_DATASETS
    source_datasets = [[FLAGS.raw_data_dir, [item[0]]] for item in datasets]
    target_datasets = [[FLAGS.raw_data_dir, [item[1]]] for item in datasets]
    # Copy vocab to data directory
    vocab_path = os.path.join(data_dir, self.vocab_name)
    if os.path.exists(vocab_path):
        os.remove(vocab_path)
    copyVocab(os.path.join(FLAGS.raw_data_dir, _VOCABS[2]), vocab_path)
    token_vocab = text_encoder.TokenTextEncoder(vocab_path, replace_oov="<unk>")
    tag = "train" if train else "dev"
    data_path = _compile_data(tmp_dir, datasets, "generic_tok_%s" % tag)
    return translate.token_generator(data_path + ".src", data_path + ".trg",
                                     token_vocab, EOS)


@registry.register_problem
class TranslateGenericExistingVocabSentenceScale(TranslateGenericExistingVocab):
  """Problem spec for generic translation, using existing vocab, with a scaling factor for each sequence"""
  
  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.VarLenFeature(tf.int64),
        "sequence_scale": tf.VarLenFeature(tf.float32),

    }
    return data_fields, None    


  def hparams(self, defaults, model_hparams):
    super(TranslateGenericExistingVocabSentenceScale, self).hparams(defaults, model_hparams)
    target_vocab_size = self._encoders["targets"].vocab_size
    defaults.target_modality = ("{}:sequence".format(registry.Modalities.SYMBOL), target_vocab_size)
    defaults.input_modality['inputs'] = ("{}:sequence".format(registry.Modalities.SYMBOL), target_vocab_size)


  def generator(self, data_dir, tmp_dir, train):
    datasets = _TRAIN_DATASETS if train else _TEST_DATASETS
    source_datasets = [[FLAGS.raw_data_dir, [item[0]]] for item in datasets]
    target_datasets = [[FLAGS.raw_data_dir, [item[1]]] for item in datasets]
    # Copy vocab to data directory
    source_vocab_path = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_path = os.path.join(data_dir, self.target_vocab_name)
    if os.path.exists(source_vocab_path):
        os.remove(source_vocab_path)
    if os.path.exists(target_vocab_path):
        os.remove(target_vocab_path)

    copyVocab(os.path.join(FLAGS.raw_data_dir, _VOCABS[0]), source_vocab_path)
    copyVocab(os.path.join(FLAGS.raw_data_dir, _VOCABS[1]), target_vocab_path)
  
    source_token_vocab = text_encoder.TokenTextEncoder(source_vocab_path, replace_oov="<unk>")
    target_token_vocab = text_encoder.TokenTextEncoder(target_vocab_path, replace_oov="<unk>")
    tag = "train" if train else "dev"
    data_path = _compile_data(tmp_dir, datasets, "generic_tok_%s" % tag)
    generator_args = ["{}.src".format(data_path),
                      "{}.trg".format(data_path),
                      source_token_vocab,
                      target_token_vocab,
                      EOS]
    scale_path = None
    if train:
      scale_path = os.path.join(FLAGS.raw_data_dir, _SCALE)
      if not os.path.isfile(scale_path):
        tf.logging.info('Generating default scale 1 for all sentences')
        scale_path = None
    generator_args.append(scale_path)
    return bi_vocabs_with_scaling_token_generator(*generator_args)


def bi_vocabs_with_scaling_token_generator(source_path,
                                           target_path,
                                           source_token_vocab,
                                           target_token_vocab,
                                           eos,
                                           scale_path):
  """Generator for sequence-to-sequence tasks that uses tokens.
  This generator assumes the files at source_path and target_path have
  the same number of lines and yields dictionaries of "inputs" and "targets"
  where inputs are token ids from the " "-split source (and target, resp.) lines
  converted to integers using the token_map.
  Args:
    source_path: path to the file with source sentences.
    target_path: path to the file with target sentences.
    source_token_vocab: text_encoder.TextEncoder object.
    target_token_vocab: text_encoder.TextEncoder object.
    eos: integer to append at the end of each sequence (default: None).
    scale_path: path to file with scaling factors for sentence pairs.
  Yields:
    A dictionary {"inputs": source-line, "targets": target-line, "scale": scaling-factor} where
    the lines are integer lists converted from tokens in the file lines.
  """
  eos_list = [eos]
  if scale_path and os.path.isfile(scale_path):
    with tf.gfile.GFile(source_path, mode="r") as source_file:
      with tf.gfile.GFile(target_path, mode="r") as target_file:
        with tf.gfile.GFile(scale_path, mode="r") as scale_file:
          source, target, scale = source_file.readline(), target_file.readline(), scale_file.readline()
          while source and target and scale:
            source_ints = source_token_vocab.encode(source.strip()) + eos_list
            target_ints = target_token_vocab.encode(target.strip()) + eos_list
            scale_float = float(scale.strip())
            yield {"inputs": source_ints, "targets": target_ints, "sequence_scale": [scale_float]}
            source, target, scale = source_file.readline(), target_file.readline(), scale_file.readline()
  else:
    with tf.gfile.GFile(source_path, mode="r") as source_file:
      with tf.gfile.GFile(target_path, mode="r") as target_file:
          source, target = source_file.readline(), target_file.readline()
          while source and target:
            source_ints = source_token_vocab.encode(source.strip()) + eos_list
            target_ints = target_token_vocab.encode(target.strip()) + eos_list
            yield {"inputs": source_ints,
                   "targets": target_ints,
                   "sequence_scale": [1.0]}
            source, target = source_file.readline(), target_file.readline()



