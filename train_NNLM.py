from NNLM import train_model
from data_reader import DataReader, get_pretrained_word_indexes
from data_reader import update_word_indexes_vocab, get_embeddings_matrix
from ngram_helper import extract_list_of_ngrams
from math import inf, pow, log

import torch
import torch.autograd as autograd

import time

# Constants
READ_LIMIT = inf
NGRAM_SIZE = 3
CONTEXT_SIZE = NGRAM_SIZE - 1
WORD_EMBEDDINGS_DIMENSION = 50

pretrained_filepath = "data/glove.6B.50d.txt"
training_data_filepath = "data/train.txt"

def save_model(model):
  torch.save(model.state_dict(), "saved_model.pt")

def main():
  # Read corpus and compile the vocabulary
  training_data = DataReader(training_data_filepath, read_limit=READ_LIMIT)
  vocab = training_data.vocab

  # Build a list of trigrams
  words = training_data.get_words()
  trigrams = extract_list_of_ngrams(words, NGRAM_SIZE)

  # Get the pretrained word vectors
  word_to_index, embed_dict = get_pretrained_word_indexes(pretrained_filepath)

  # Update word_to_index and vocabulary
  word_to_index, vocab = update_word_indexes_vocab(word_to_index, vocab)

  # Get the numpy matrix containing the pretrained word vectors
  # with randomly initialized unknown words from the corpus
  word_embeddings = get_embeddings_matrix(word_to_index, embed_dict, WORD_EMBEDDINGS_DIMENSION)

  print("Done reading data and creating trigrams")

  start_time = time.time()

  trained_model = train_model(trigrams, len(vocab), CONTEXT_SIZE, word_to_index, \
      word_embeddings, WORD_EMBEDDINGS_DIMENSION)
  save_model(trained_model)

  print("Final training time: ", time.time() - start_time)

main()
