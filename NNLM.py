import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_reader import DataReader
from ngram_helper import extract_list_of_ngrams

word_vector_dimension = 50

training_data = DataReader("data/penn/train.txt", read_limit=10)

word_to_index = training_data.get_word_indexes()
vocab_size = len(word_to_index)

embeds = nn.Embedding(vocab_size, word_vector_dimension)

words = training_data.get_words()
trigrams = extract_list_of_ngrams(words, 3)

for trigram in trigrams: print(trigram)
