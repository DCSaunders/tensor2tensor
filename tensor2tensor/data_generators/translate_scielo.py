# coding=utf-8
"""Additional problems for standard T2T models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators.translate_ende import TranslateEndeWmt32k
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_encoder
import os

# Start CF project
@registry.register_problem
class TranslateEsenScieloHealth(TranslateEndeWmt32k):
  @property
  def vocab_file(self):
    return "vocab.esen"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(vocab_filename)}

@registry.register_problem
class TranslateEsenScieloBio(TranslateEndeWmt32k):
  @property
  def vocab_file(self):
    return "vocab.esen"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(vocab_filename)}

@registry.register_problem
class TranslateEsenScieloHealthBio(TranslateEndeWmt32k):
  @property
  def vocab_file(self):
    return "vocab.esen"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(vocab_filename)}

@registry.register_problem
class TranslateEsenScieloBioAndHealth(TranslateEndeWmt32k):
  @property
  def vocab_file(self):
    return "vocab.esen"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(vocab_filename)}
# End CF project


@registry.register_problem
class TranslateJaenKyoto32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

