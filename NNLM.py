import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_reader import DataReader

word_vector_dimention = 50

traing_data = DataReader("data/penn/train.txt")

word_to_index = traing_data.get_word_indexes()
vocab_size = len(word_to_index)

print(vocab_size)
