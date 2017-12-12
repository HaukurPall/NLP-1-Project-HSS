import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn._functions.rnn import Recurrent, StackedRNN
from torch.autograd import Variable

class RAN(nn.Module):

    def __init__(self, input_size, vocab_size, word_embeddings, use_GPU, nlayers=1, dropout=0.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = input_size
        self.nlayers = nlayers
        self.dropout = dropout
        self.use_GPU = use_GPU

        # self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.embeddings.weight.data.copy_(torch.from_numpy(word_embeddings))
        # self.embeddings.weight.requires_grad = False # Do not train the pre-calculated embeddings

        self.linear = nn.Linear(self.hidden_size, vocab_size)
        # self.linear.weight.data.copy_(word_embeddings)
        # self.linear.weight.requires_grad = False

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
        layer = (Recurrent(RANCell), )
        func = StackedRNN(layer, self.nlayers, dropout=self.dropout)
        hidden, output = func(input, hidden, ((self.weights, self.biases), ))

        if self.use_GPU:
            hidden, output = hidden.cuda(), output.cuda()

        output = F.log_softmax(self.linear(output))

        return output, hidden

    def init_hidden(self):
        variable = Variable(torch.zeros(1, self.hidden_size))
        return variable if not self.use_GPU else variable.cuda()

def RANCell(input, state, weights, biases):
    w_ic, w_ix, w_fc, w_fx = weights
    b_i, b_f = biases
    # OBS: we are not sure if we have to pass in a bias or not
    i_t = F.sigmoid(F.linear(state, w_ic, b_i) + F.linear(input, w_ix))
    f_t = F.sigmoid(F.linear(state, w_fc, b_f) + F.linear(input, w_fx))
    c_t = i_t * input + f_t * state

    return c_t
