import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn._functions.rnn import Recurrent, StackedRNN
from torch.autograd import Variable

class RAN(nn.Module):
    def __init__(self, \
                 vocab_size, \
                 word_embeddings_dimension, \
                 num_layers, \
                 dropout_prob, \
                 word_embeddings, \
                 use_pretrained=True, \
                 use_GPU=True):
        super().__init__()
        self.input_size = vocab_size
        self.hidden_size = word_embeddings_dimension
        self.dropout = nn.Dropout(dropout_prob)
        self.use_GPU = use_GPU
        self.num_layers = num_layers

        self.encoder = nn.Embedding(self.input_size, self.hidden_size)
        if use_pretrained:
            # we use the pretrained word embeddings
            self.encoder.weight.data.copy_(word_embeddings)

        self.encoder.weight.requires_grad = False # Do not train the embeddings

        self.decoder = nn.Linear(self.hidden_size, self.input_size, bias=True)
        self.decoder.weight = self.encoder.weight
        # do not train the
        self.decoder.weight.requires_grad = True

        if use_GPU:
            self.w_ic = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size).cuda())
            self.w_ix = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size).cuda())
            self.w_fc = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size).cuda())
            self.w_fx = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size).cuda())

            self.b_i = nn.Parameter(torch.Tensor(self.hidden_size).cuda())
            self.b_f = nn.Parameter(torch.Tensor(self.hidden_size).cuda())

        else:
            self.w_ic = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
            self.w_ix = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
            self.w_fc = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
            self.w_fx = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))

            self.b_i = nn.Parameter(torch.Tensor(self.hidden_size))
            self.b_f = nn.Parameter(torch.Tensor(self.hidden_size))

        self.weights = self.w_ic, self.w_ix, self.w_fc, self.w_fx
        for w in self.weights:
            init.xavier_uniform(w)

        if not use_pretrained:
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.weight.data.uniform_(-initrange, initrange)

        self.biases = self.b_i, self.b_f
        for b in self.biases:
            b.data.fill_(0)

    def forward(self, input, hidden):
        # apply dropout to word_embeddings
        input = self.dropout(self.encoder(input))
        layer = (Recurrent(RANCell), )
        # we implicitily set the layers to 1
        func = StackedRNN(layer, self.num_layers, dropout=self.dropout)
        # we apply the RAN cell
        hidden, output = func(input, hidden, ((self.weights, self.biases), ))
        # apply dropout to output
        output = self.dropout(output)
        if self.use_GPU:
            hidden, output = hidden.cuda(), output.cuda()
        # we decode (from hidden to word_embeddings)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        # and return the output nicely formatted for the loss function
        decoded_formatted = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return decoded_formatted, hidden

    def init_hidden(self, batch_size):
        # implicitily set the layers to 1
        variable = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        return variable if not self.use_GPU else variable.cuda()

def RANCell(input, state, weights, biases):
    w_ic, w_ix, w_fc, w_fx = weights
    b_i, b_f = biases
    # OBS: we are not sure if we have to pass in a bias or not
    i_t = F.sigmoid(F.linear(state, w_ic, b_i) + F.linear(input, w_ix))
    f_t = F.sigmoid(F.linear(state, w_fc, b_f) + F.linear(input, w_fx))
    c_t = i_t * input + f_t * state

    return c_t
