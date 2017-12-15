import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, \
                 vocab_size, \
                 word_embeddings_dimension, \
                 num_layers, \
                 dropout_prob, \
                 word_embeddings, \
                 use_pretrained=True, \
                 use_GPU=True):
        super(LSTM, self).__init__()
        self.input_size = vocab_size
        self.hidden_size = word_embeddings_dimension
        self.num_layers = num_layers

        self.drop = nn.Dropout(dropout_prob)
        self.encoder = nn.Embedding(self.input_size, self.hidden_size)

        if use_pretrained:
            self.encoder.weight.data.copy_(word_embeddings)

        self.encoder.weight.requires_grad = False # Do not train the embeddings

        self.rnn = getattr(nn, 'LSTM')(self.hidden_size, \
                                       self.hidden_size, \
                                       num_layers, \
                                       dropout=dropout)
        self.decoder = nn.Linear(self.hidden_size, self.input_size, bias=False)
        self.decoder.weight = self.encoder.weight
        # do not train the
        self.decoder.weight.requires_grad = True

        # if tie_weights:
        #     if hidden_size != word_embeddings_dimension:
        #         raise ValueError('When using the tied flag, nhid must be equal to emsize')
        #     self.decoder.weight = self.encoder.weight
        #     self.decoder.weight.requires_grad = True

        if not use_pretrained:
            self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, batch_size, self.num_hidden).zero_()),
                    Variable(weight.new(self.num_layers, batch_size, self.num_hidden).zero_()))

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
