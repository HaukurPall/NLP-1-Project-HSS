from collections import defaultdict
from math import inf
import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np

torch.manual_seed(1)

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

def get_pretrained_word_indexes(filepath=""):
  word_to_index = {}
  word_embeddings = {}

  with open(filepath, "r") as f:
    for line_number, line in enumerate(f.readlines()):
        words = line.split()
        word = words[0]
        embed = list(map(float, words[1:]))
        # if line_number == 1:
            # print(embed)
            # print(embed)
            # print(len(embed))
        word_to_index[word] = line_number
        word_embeddings[word] = embed

  return word_to_index, word_embeddings

def update_word_indexes_vocab(word_to_index, vocab):

    for word in vocab:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

    vocab = word_to_index.keys()

    return word_to_index, vocab

def get_embeddings_matrix(word_to_index, embed_dict, embed_dimension):
    pretrained_weight = np.random.uniform(-1, 1, (len(word_to_index), embed_dimension))
    for word in word_to_index.keys():
        if word in embed_dict:
            pretrained_weight[word_to_index[word]] = embed_dict[word]

    return pretrained_weight

# From Stian
# def get_pretrained_word_indexes():
#   word_to_index = {}
#
#   with open(pretrained_filepath, "r") as f:
#     for line_number, line in enumerate(f.readlines()):
#       if line_number != 0: # Skip the first line since it contains length and dimension
#         word = line.split()[0]
#         word_to_index[word] = line_number - 1
#
#   return word_to_index
#
# def get_pretrained_embeddings():
#   vocab, vec = torchwordemb.load_word2vec_text(pretrained_filepath)
#   return vocab, vec