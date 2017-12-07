import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd

from data_reader import DataReader, get_pretrained_word_indexes
from data_reader import update_word_indexes_vocab, get_embeddings_matrix

###### Constants ######
N_HIDDEN = 128
WORD_EMBEDDINGS_DIMENSION = 50

training_data_filepath = "data/train.txt"
# pretrained_embeddings_filepath = "data/glove.6B.50d.txt"
#
# ###### Read data ######
# training_data = DataReader(training_data_filepath)
# vocab = training_data.vocab
#
# # Get the pretrained word vectors
# word_to_index, embed_dict = get_pretrained_word_indexes(pretrained_embeddings_filepath)
#
# # Update word_to_index and vocabulary
# word_to_index, vocab = update_word_indexes_vocab(word_to_index, vocab)
#
# # Get the numpy matrix containing the pretrained word vectors
# # with randomly initialized unknown words from the corpus
# word_embeddings = get_embeddings_matrix(word_to_index, embed_dict, WORD_EMBEDDINGS_DIMENSION)
#
# # Make a torch that contains the embeddings for all the words in the vocabulary
# embeddings_torch = torch.from_numpy(word_embeddings)

training_data = DataReader(training_data_filepath)
vocab = training_data.get_vocabulary()
vocab_size = len(vocab)

word_to_index = {word : i for i, word in enumerate(vocab)}

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        output_size = input_size

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

def word_to_tensor(word):
    # One hot
    word_index = word_to_index[word]
    word_vector = torch.zeros(vocab_size, 1)
    print(word_vector)
    word_vector[word_index] = 1
    return word_vector

rnn = RNN(len(vocab), N_HIDDEN)

input = Variable(word_to_tensor("the"))
print(input)
initial_hidden = Variable(torch.zeros(1, N_HIDDEN))
