import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from RAN import RAN

from data_reader import DataReader, get_pretrained_word_indexes
from data_reader import update_word_indexes_vocab, get_embeddings_matrix
from ngram_helper import extract_list_of_ngrams

from math import inf, exp
from collections import defaultdict
from datetime import datetime
import time

#### Options

use_GPU = True

#### Constants

EPOCHS = 100
BATCH_SIZE = 20
EVAL_BATCH_SIZE = BATCH_SIZE
CONTEXT_SIZE = 30
WORD_EMBEDDINGS_DIMENSION = 50
LEARNING_RATE = 1
LOSS_CLIP = 30

BATCH_LOG_INTERVAL = 50
READ_LIMIT = inf

pretrained_embeddings_filepath = "data/glove.6B.{}d.txt".format(WORD_EMBEDDINGS_DIMENSION)
training_data_filepath = "data/train.txt"
validation_data_filepath = "data/valid.txt"

timestamp = str(datetime.now()).split()[1][:8].replace(":", "_")

timestamp_signature = "{}_{}_batch_{:d}_embed_{}_learn_{}".format("RAN", timestamp, BATCH_SIZE, WORD_EMBEDDINGS_DIMENSION, str(LEARNING_RATE)[:4])
perplexity_filepath = "perplexities/" + timestamp_signature + ".txt"
print(perplexity_filepath)

def prepare_dictionaries(training_data):
    vocab = training_data.get_vocabulary()
    word_to_index, index_to_word = training_data.get_word_to_index_to_word()

    # Get the pretrained word vectors and build a word_to_index based on it
    _, embed_dict = get_pretrained_word_indexes(pretrained_embeddings_filepath)

    # Get the numpy matrix containing the pretrained word vectors
    # with randomly initialized unknown words for words that do not occur in the pretrained
    # embeddings, but that occur in the corpus (roughly 73 words)
    word_embeddings = get_embeddings_matrix(word_to_index, embed_dict, WORD_EMBEDDINGS_DIMENSION)

    # Convert the numpy matrix to a torch tensor
    word_embeddings = torch.from_numpy(word_embeddings)
    return word_to_index, index_to_word, word_embeddings, len(vocab)

###### Read in all the data outside of functions so we can refer to them 'globally' #####
# Read corpus and compile the vocabulary
training_data = DataReader(training_data_filepath, read_limit=READ_LIMIT)

word_to_index, index_to_word, word_embeddings, vocab_size = prepare_dictionaries(training_data)
training_words = training_data.get_words()
# now the training data is a one dimensional vector of indexes
training_data = torch.LongTensor([word_to_index[word] for word in training_words])

###### Same operations for the validation dataset ######

# When we have unseen words in the validation set we default to unknown
word_to_index = defaultdict(lambda: word_to_index["unknown"], word_to_index)

validation_data = DataReader(validation_data_filepath, read_limit=READ_LIMIT)

validation_words = validation_data.get_words()
# now the training data is a one dimensional vector of indexes
validation_data = torch.LongTensor([word_to_index[word] for word in validation_words])

print("### Done reading data ###")

# We change the dataset from a one dimensional vector to a x*batch_size matrix
def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(batch_size, -1).t().contiguous()
    if use_GPU:
        data = data.cuda()
    return data

training_data = batchify(training_data, BATCH_SIZE)
validation_data = batchify(validation_data, BATCH_SIZE)

def save_model(model, epoch):
    torch.save(model.state_dict(), "saved_models/" + timestamp_signature + str(epoch) + ".pt")

def get_batch(source, i, evaluation=False):
    seq_len = min(CONTEXT_SIZE, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    if use_GPU:
      data, target = data.cuda(), target.cuda()
    return data, target

def evaluate(data_source, ran, criterion):
    # Turn on evaluation mode which disables dropout.
    ran.eval()
    total_loss = 0
    for i in range(0, data_source.size(0) - 1, EVAL_BATCH_SIZE):
        hidden = ran.init_hidden(EVAL_BATCH_SIZE)
        data, targets = get_batch(data_source, i, evaluation=True)

        output, hidden = ran(data, hidden)
        output_flat = output.view(-1, vocab_size)
        total_loss += len(data) * criterion(output_flat, targets).data

    return total_loss[0] / len(data_source)

def save_perplexity(filepath, perplexity, epoch):
    with open(filepath, "a") as f:
        f.write(str(epoch) + " " + str(perplexity) + "\n")

    return True

def train_RAN(training_data, learning_rate, epochs, vocab_size, word_embeddings, use_GPU):
    criterion = nn.CrossEntropyLoss() if not use_GPU else nn.CrossEntropyLoss().cuda()

    ran = RAN(WORD_EMBEDDINGS_DIMENSION, word_embeddings, use_GPU)

    if use_GPU:
        ran = ran.cuda()

    start_time = time.time()

    for epoch in range(epochs):
        # learning_rate *= 0.95 # Reduce learning rate each epoch
        if epoch > 6:
          learning_rate /= 1.2

        # turn on dropouts
        total_loss = 0
        ran.train()

        for batch, i in enumerate(range(0, training_data.size(0) - 1, CONTEXT_SIZE)):
            ran.zero_grad()
            hidden = ran.init_hidden(BATCH_SIZE)

            data, targets = get_batch(training_data, i)

            output, hidden = ran(data, hidden)

            loss = criterion(output.view(-1, vocab_size), targets)

            loss.backward(retain_graph=True) # Haukur: I set it to False, want to hear the reasoning why it was set. Stian: Don't understand why we need this argument

            # # Clip gradients to avoid gradient explosion
            # torch.nn.utils.clip_grad_norm(ran.parameters(), LOSS_CLIP)
            for p in ran.parameters():
                if p.requires_grad:
                    p.data.add_(-learning_rate, p.grad.data)

            total_loss += loss.data

            if batch % BATCH_LOG_INTERVAL == 0 and batch > 0:
                print("Epoch: ", epoch)
                print("batch", batch, "out of ", training_data.size(0) // BATCH_SIZE)
                cur_loss = total_loss[0] / BATCH_LOG_INTERVAL
                print("Loss", loss)
                print("current perplexity:", exp(cur_loss))
                total_loss = 0

        save_model(ran, epoch)
        average_loss = evaluate(validation_data, ran, criterion)
        validation_perplexity = exp(average_loss)

        print("Validation perplexity", validation_perplexity, "Loss", average_loss)
        save_perplexity(perplexity_filepath, validation_perplexity, epoch)

train_RAN(training_data=training_data, \
          learning_rate=LEARNING_RATE, \
          epochs=EPOCHS, \
          vocab_size=vocab_size, \
          word_embeddings=word_embeddings, \
          use_GPU=use_GPU)
