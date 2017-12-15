import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from data_reader import DataReader, get_pretrained_word_indexes
from data_reader import update_word_indexes_vocab, get_embeddings_matrix, convert_long_tensor
from ngram_helper import extract_list_of_ngrams
from math import inf, exp
import time
from collections import defaultdict
from datetime import datetime
from model_LSTM import LSTMModel

use_GPU = True
print("Using GPU" if use_GPU else "Running without GPU")

use_pretrained = True

# Constants
READ_LIMIT = inf # Manually reset if we want faster processing
EMBEDDING_DIM = 50
NUM_HIDDEN_UNITS = EMBEDDING_DIM
NUM_LAYERS = 1
DROPOUT_PROB = 0.2
BATCH_SIZE = 64
EVAL_BATCH_SIZE = BATCH_SIZE
SEQ_LENGTH = 35
NUM_EPOCHS = 100
# LEARNING_RATE = 1
LEARNING_RATE = 5
MIN_LEARNING_RATE = 0.001
BATCH_LOG_INTERVAL = 100
LOSS_CLIP = 10

# File paths
embedding_path = "data/glove.6B.{}d.txt".format(EMBEDDING_DIM)
train_data_path = "data/train.txt"
valid_data_path = "data/valid.txt"

timestamp = str(datetime.now()).split()[1][:8].replace(":", "_")
timestamp_signature = "{}_{}_batch_{:d}_embed_{}_learn_{}".format("LSTM", timestamp, BATCH_SIZE, EMBEDDING_DIM, str(LEARNING_RATE)[:4])
perplexity_filepath = "perplexities/" + timestamp_signature + ".txt"

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
    word_embeddings = get_embeddings_matrix(word_to_ix, embed_dict, EMBEDDING_DIM)

words_tensor = torch.LongTensor([word_to_ix[word] for word in words])
train_data = batchify(words_tensor, BATCH_SIZE)

# Validation data
valid_word_to_ix = defaultdict(lambda: word_to_ix["unknown"], word_to_ix)

validation_data = DataReader(valid_data_path, read_limit=READ_LIMIT)
validation_words = validation_data.get_words()
# now the training data is a one dimensional vector of indexes
validation_words_tensor = torch.LongTensor([valid_word_to_ix[word] for word in validation_words])
valid_data = batchify(validation_words_tensor, BATCH_SIZE)

# Build RNN/LSTM model
lstm = LSTMModel(vocab_size, EMBEDDING_DIM, NUM_HIDDEN_UNITS, NUM_LAYERS, DROPOUT_PROB, word_embeddings, use_pretrained, tie_weights=True)
if use_GPU:
    lstm.cuda()

loss_function = nn.CrossEntropyLoss()

def evaluate(data_source, lstm, loss_function):
    # Turn on evaluation mode which disables dropout.
    lstm.eval()
    total_loss = 0
    hidden = lstm.init_hidden(EVAL_BATCH_SIZE)
    for i in range(0, data_source.size(0) - 1, SEQ_LENGTH):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = lstm(data, hidden)
        output_flat = output.view(-1, vocab_size)
        total_loss += len(data) * loss_function(output_flat, targets).data
        hidden = repackage_hidden(hidden)

    return total_loss[0] / len(data_source)

def save_perplexity(filepath, perplexity, epoch):
    with open(filepath, "a") as f:
        f.write(str(epoch) + " " + str(perplexity) + "\n")
    return True

def save_model(model, epoch):
    torch.save(model.state_dict(), "saved_models/" + timestamp_signature + str(epoch) + ".pt")

def has_improved(checkpoint_perplexities):
    if len(checkpoint_perplexities) < 30:
        # We don't want to reduce the learning rate if have not made 30 checkpoints yet
        return True

    print("Checkpoint values", checkpoint_perplexities[-30],  checkpoint_perplexities[-1])

    decreases = 0

    for i in range(len(checkpoint_perplexities) - 30, len(checkpoint_perplexities)):
        if checkpoint_perplexities[i] - checkpoint_perplexities[i-1] < 0:
            decreases += 1

    print("decreases:", decreases)
    return decreases >= 20

def train():

    checkpoint_counter = 0
    checkpoint_perplexities = []

    learning_rate = LEARNING_RATE
    for epoch in range(1, NUM_EPOCHS +1):
        # Turn on training mode which enables dropout.
        lstm.train()
        total_loss = 0

        start_time = time.time()
        hidden = lstm.init_hidden(BATCH_SIZE)

        for batch, i in enumerate(range(0, train_data.size(0) - 1, SEQ_LENGTH)):
            data, targets = get_batch(train_data, i)

            hidden = repackage_hidden(hidden)
            lstm.zero_grad()
            output, hidden = lstm(data, hidden)

            loss = loss_function(output.view(-1, vocab_size), targets)
            loss.backward()

            checkpoint_counter += 1
            if checkpoint_counter == 100:
                checkpoint_counter = 0
                checkpoint_perplexities.append(exp(evaluate(valid_data, lstm, loss_function)))
                if not has_improved(checkpoint_perplexities):
                    learning_rate = max(learning_rate*0.75, MIN_LEARNING_RATE)
                    print("Reduced learning rate to", learning_rate)

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)

            # Clip gradients to avoid gradient explosion
            torch.nn.utils.clip_grad_norm(lstm.parameters(), LOSS_CLIP)
            for p in lstm.parameters():
                if p.requires_grad:
                    p.data.add_(-learning_rate, p.grad.data)

            total_loss += loss.data

            if batch % BATCH_LOG_INTERVAL == 0 and batch > 0:
                cur_loss = total_loss[0] / BATCH_LOG_INTERVAL
                total_loss = 0

        save_model(lstm, epoch)
        average_loss = evaluate(valid_data, lstm, loss_function)
        validation_perplexity = exp(average_loss)

        print("\nEpoch", epoch, "Validation perplexity", \
                validation_perplexity, "learning_rate:", learning_rate, "\n")

        save_perplexity(perplexity_filepath, validation_perplexity, epoch)

train()
