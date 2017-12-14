import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from data_reader import DataReader, get_pretrained_word_indexes
from data_reader import update_word_indexes_vocab, get_embeddings_matrix, convert_long_tensor
from ngram_helper import extract_list_of_ngrams
from math import inf
import time
from model_LSTM import LSTMModel

use_GPU = False
use_pretrained = True

# Constants
READ_LIMIT = inf # Manually reset if we want faster processing
EMBEDDING_DIM = 50
NUM_HIDDEN_UNITS = 50
NUM_LAYERS = 1
DROPOUT_PROB = 0.2
BATCH_SIZE = 10
SEQ_LENGTH = 30 # Default: Trigram
NUM_EPOCHS = 2
LEARNING_RATE = 0.001

# File paths
embedding_path = "data/glove.6B.50d.txt"
train_data_path = "data/train.txt"
valid_data_path = "data/valid.txt"


torch.manual_seed(1111)
if use_GPU:
    torch.cuda.manual_seed(1111)

def batchify(data, batch_size=10):
    # Work out how cleanly we can divide the dataset into batch_size parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    if use_GPU:
        data = data.cuda()
    return data

def get_batch(source, i, evaluation=False):
    seq_len = min(SEQ_LENGTH, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

# Read corpus and compile the vocabulary
training_data = DataReader(train_data_path, read_limit=READ_LIMIT)
vocab = training_data.get_vocabulary()
vocab_size = len(vocab)
words = training_data.get_words()
words_size = len(words)
word_embeddings = None

word_to_ix, _ = training_data.get_word_to_index_to_word()
if use_pretrained:
    _, embed_dict = get_pretrained_word_indexes(embedding_path)
    # word_to_ix, vocab = update_word_indexes_vocab(word_to_ix, vocab)
    # vocab_size = len(vocab)
    word_embeddings = get_embeddings_matrix(word_to_ix, embed_dict, EMBEDDING_DIM)
# else:
#     word_to_ix = training_data.get_word_indexes()

# print('vocab size',vocab_size)
# print('embed',torch.from_numpy(word_embeddings).size(0))
# Convert the training_data into Long Tensors
words_tensor =  convert_long_tensor(words, word_to_ix, words_size)
# print('words_tensor', words_tensor)
train_data = batchify(words_tensor, BATCH_SIZE)
# print('train_data', train_data)

# Build RNN/LSTM model
model = LSTMModel(vocab_size, EMBEDDING_DIM, NUM_HIDDEN_UNITS, NUM_LAYERS, DROPOUT_PROB, word_embeddings, use_pretrained)
if use_GPU:
    model.cuda()

loss_function = nn.CrossEntropyLoss()


def train():

    for epoch in range(1, NUM_EPOCHS +1):
        # Turn on training mode which enables dropout.
        lr = LEARNING_RATE
        model.train()
        total_loss = 0
        start_time = time.time()
        hidden = model.init_hidden(BATCH_SIZE)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, SEQ_LENGTH)):
            # print('batch',batch)
            # print('len source',train_data.size(0))
            data, targets = get_batch(train_data, i)
            # print('data',data)
            # print('targets',targets)
            # break
            hidden = repackage_hidden(hidden)
            model.zero_grad()
            output, hidden = model(data, hidden)
            # print('output', output)
            # print('output view', output.view(-1, vocab_size))
            loss = loss_function(output.view(-1, vocab_size), targets)
            # print('loss',loss)
            # break
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            for p in model.parameters():
                if p.requires_grad:
                    p.data.add_(-lr, p.grad.data)

            total_loss += loss.data

            if batch % 200 == 0 and batch > 0:
                cur_loss = total_loss[0] / 200
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_data) // SEQ_LENGTH, lr,
                        elapsed * 1000 / 200, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

        torch.save(lstm, epoch)

train()
