from collections import defaultdict
from math import inf
import torchwordemb

pretrained_filepath = "data/standford/correct_format.txt"

class DataReader:
  def __init__(self, filepath="", read_limit=inf):
    self.words = []
    self.sentences = []
    if filepath:
      with open(filepath, "r") as f:
        for line_number, line in enumerate(f.readlines()):
          line = replace_unk_with_unknown(line)
          self.sentences.append(line)
          for word in line.split():
            self.words.append(word)
          if line_number > read_limit: break

    self.vocab = set(self.words)

  def get_words(self):
    return self.words

  def get_vocabulary(self):
    return self.vocab

  def get_sentences(self):
    return self.sentences

  def get_word_indexes(self):
    # Returns a dictionary containing all the words in the vocabulary
    # and their respective indexes.
    word_to_index = defaultdict(lambda: len(word_to_index))

    for word in self.words:
      word_to_index[word]

    return word_to_index

def replace_unk_with_unknown(sentence):
  return sentence.replace("<unk>", "unknown")

def get_pretrained_word_indexes():
  word_to_index = {}

  with open(pretrained_filepath, "r") as f:
    for line_number, line in enumerate(f.readlines()):
      if line_number != 0: # Skip the first line since it contains length and dimension
        word = line.split()[0]
        word_to_index[word] = line_number - 1

  return word_to_index

def get_pretrained_embeddings():
  vocab, vec = torchwordemb.load_word2vec_text(pretrained_filepath)
  return vocab, vec
