import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn._functions.rnn import Recurrent, StackedRNN
from torch.autograd import Variable

class RAN(nn.Module):

    def __init__(self, input_size, word_embeddings, use_GPU, dropout=0.5):
        super().__init__()
        self.input_size = input_size
        # we use the same size for input and hidden
        self.hidden_size = input_size
        self.dropout = nn.Dropout(dropout)
        self.use_GPU = use_GPU
        vocab_size = word_embeddings.size(0)
        embeddings_dimension = word_embeddings.size(1)

        # we use the pretrained word embeddings
        self.embeddings = nn.Embedding(vocab_size, embeddings_dimension)
        self.embeddings.weight.data.copy_(word_embeddings)
        self.embeddings.weight.requires_grad = False # Do not train the pre-calculated embeddings

        self.linear = nn.Linear(self.hidden_size, vocab_size, bias=False)
        self.linear.weight = self.embeddings.weight
        # do not train the
        self.linear.weight.requires_grad = True

        if use_GPU:
            self.w_ic = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size).cuda())
            self.w_ix = nn.Parameter(torch.Tensor(self.hidden_size, input_size).cuda())
            self.w_fc = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size).cuda())
            self.w_fx = nn.Parameter(torch.Tensor(self.hidden_size, input_size).cuda())

            self.b_i = nn.Parameter(torch.Tensor(self.hidden_size).cuda())
            self.b_f = nn.Parameter(torch.Tensor(self.hidden_size).cuda())

        else:
            self.w_ic = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
            self.w_ix = nn.Parameter(torch.Tensor(self.hidden_size, input_size))
            self.w_fc = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
            self.w_fx = nn.Parameter(torch.Tensor(self.hidden_size, input_size))

            self.b_i = nn.Parameter(torch.Tensor(self.hidden_size))
            self.b_f = nn.Parameter(torch.Tensor(self.hidden_size))

        self.weights = self.w_ic, self.w_ix, self.w_fc, self.w_fx
        for w in self.weights:
            init.xavier_uniform(w)

        self.biases = self.b_i, self.b_f
        for b in self.biases:
            b.data.fill_(0)

    def forward(self, input, hidden):
        # apply dropout to word_embeddings
        input = self.dropout(self.embeddings(input))
        layer = (Recurrent(RANCell), )
        # we implicitily set the layers to 1
        func = StackedRNN(layer, 1, dropout=self.dropout)
        # we apply the RAN cell
        hidden, output = func(input, hidden, ((self.weights, self.biases), ))
        # apply dropout to output
        output = self.dropout(output)
        if self.use_GPU:
            hidden, output = hidden.cuda(), output.cuda()
        # we decode (from hidden to word_embeddings)
        decoded = self.linear(output.view(output.size(0)*output.size(1), output.size(2)))
        # and return the output nicely formatted for the loss function
        decoded_formatted = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return decoded_formatted, hidden

    def init_hidden(self, batch_size):
        # implicitily set the layers to 1
        variable = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return variable if not self.use_GPU else variable.cuda()

def RANCell(input, state, weights, biases):
    w_ic, w_ix, w_fc, w_fx = weights
    b_i, b_f = biases
    # OBS: we are not sure if we have to pass in a bias or not
    i_t = F.sigmoid(F.linear(state, w_ic, b_i) + F.linear(input, w_ix))
    f_t = F.sigmoid(F.linear(state, w_fc, b_f) + F.linear(input, w_fx))
    c_t = i_t * input + f_t * state

    return c_t
