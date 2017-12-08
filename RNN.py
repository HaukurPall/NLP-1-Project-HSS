import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd

from data_reader import DataReader, get_pretrained_word_indexes
from data_reader import update_word_indexes_vocab, get_embeddings_matrix
from ngram_helper import extract_list_of_ngrams

###### Constants ######
N_HIDDEN = 128
WORD_EMBEDDINGS_DIMENSION = 50
LEARNING_RATE = 0.05
CONTEXT_SIZE = 4

training_data_filepath = "data/train.txt"

training_data = DataReader(training_data_filepath)
vocab = training_data.get_vocabulary()
vocab_size = len(vocab)

words = training_data.get_words()
ngrams = extract_list_of_ngrams(words, CONTEXT_SIZE - 1)

word_to_index = {word : i for i, word in enumerate(vocab)}
index_to_word = {i : word for word, i in word_to_index.items()}

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
    # Create a one-hot-vector representing the given word
    word_index = word_to_index[word]
    word_vector = torch.zeros(vocab_size, 1)
    word_vector[word_index] = 1
    return word_vector.view(1, -1)

def word_from_output(output):
    top_n, top_i = output.data.topk(1)
    word_index = top_i[0][0]

    return index_to_word[word_index], word_index

rnn = RNN(len(vocab), N_HIDDEN)
criterion = nn.NLLLoss()

def get_training_example(i):
    context = ngrams[i][0]
    target_word = ngrams[i][1]
    target_variable = Variable(torch.LongTensor([word_to_index[target_word]]))
    context_tensor = torch.zeros((CONTEXT_SIZE, vocab_size))

    for i in range(len(context)):
        context_tensor[i] = word_to_tensor(context[i])

    context_variable = Variable(context_tensor)

    return target_word, context, target_variable, context_variable

def train(target_tensor, context_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(context_tensor.size()[0]):
        output, hidden = rnn(context_tensor[i].view(1, -1), hidden)

    print(output)
    print(target_tensor)

    loss = criterion(output, target_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-LEARNING_RATE, p.grad.data)

    return output, loss.data[0]

for i in range(len(ngrams)):
    target_word, context, target_variable, context_variable = get_training_example(i)
    output, loss = train(target_variable, context_variable)

    print("Target: ", target_word, "Context: ", context, "Output word: ", word_from_output(output))
