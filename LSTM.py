import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, vocab_size, embedding_dim, num_hidden, num_layers, dropout=0.5, word_embeddings=None, use_pretrained=False, tie_weights=False):
        super(LSTM, self).__init__()
        self.drop = nn.Dropout(dropout)
        if use_pretrained:
            self.encoder = nn.Embedding(vocab_size, embedding_dim)
            self.encoder.weight.data.copy_(torch.from_numpy(word_embeddings))
            self.encoder.weight.requires_grad = False # Do not train the pre-calculated embeddings
        else:
            self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = getattr(nn, 'LSTM')(embedding_dim, num_hidden, num_layers, dropout=dropout)
        self.decoder = nn.Linear(num_hidden, vocab_size)

        if tie_weights:
            if num_hidden != embedding_dim:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
            self.decoder.weight.requires_grad = True

        if not use_pretrained:
            self.init_weights()
        self.num_hidden = num_hidden
        self.num_layers = num_layers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
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
