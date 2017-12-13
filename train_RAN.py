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
import time

#### Options

use_GPU = True

#### Constants

EPOCHS = 10
BATCH_SIZE = 256
CONTEXT_SIZE = 30
WORD_EMBEDDINGS_DIMENSION = 50
LEARNING_RATE = 0.01
LOSS_CLIP = 30

BATCH_LOG_INTERVAL = 10
READ_LIMIT = inf

pretrained_embeddings_filepath = "data/glove.6B.50d.txt"
training_data_filepath = "data/train.txt"
validation_data_filepath = "data/valid.txt"

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

def save_model(model, epoch):
    torch.save(model.state_dict(), str(epoch) + "_saved_model_ran.pt")

def context_to_variable(context):
    context_tensor = torch.FloatTensor(CONTEXT_SIZE, WORD_EMBEDDINGS_DIMENSION)
    for i in range(len(context)):
        context_tensor[i] = word_embeddings[word_to_index[context[i]]]

    variable = Variable(context_tensor.view(1, -1))
    return variable if not use_GPU else variable.cuda()

def word_to_variable(word): # Might not be necessary
    word_tensor = word_embeddings[word_to_index[word]]
    variable = Variable(word_tensor)
    return variable if not use_GPU else variable.cuda()

def word_from_output(output):
    top_n, top_i = output.data.topk(1)
    word_index = top_i[0][0]

    return index_to_word[word_index], word_index

def print_info(i, context, target_word, outputted_word, loss):
    print(i)
    print("Context: ", context, "Target word was: ", target_word)
    print("Predicted word was: ", outputted_word)
    print("Loss: ", loss)
    if loss.data[0] == "nan":
        print("Gradient exploded...")

    if target_word == outputted_word[0]:
        print("####### Prediction was right!! ", context, target_word)

def get_batch(source, i, evaluation=False):
    seq_len = min(CONTEXT_SIZE, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def train_RAN(training_data, learning_rate, epochs, vocab_size, word_embeddings, use_GPU):
    criterion = nn.CrossEntropyLoss() if not use_GPU else nn.CrossEntropyLoss().cuda()

    ran = RAN(WORD_EMBEDDINGS_DIMENSION, word_embeddings, use_GPU)

    if use_GPU:
        ran = ran.cuda()

    start_time = time.time()

    for epoch in range(epochs):
        # learning_rate *= 0.95 # Reduce learning rate each epoch

        # turn on dropouts
        total_loss = 0
        ran.train()

        for batch, i in enumerate(range(0, training_data.size(0) - 1, CONTEXT_SIZE)):
            ran.zero_grad()
            hidden = ran.init_hidden(BATCH_SIZE)

            data, targets = get_batch(training_data, i)

            output, hidden = ran(data, hidden)

            loss = criterion(output.view(-1, vocab_size), targets)
            print(loss)

            loss.backward(retain_graph=True) # Haukur: I set it to False, want to hear the reasoning why it was set. Stian: Don't understand why we need this argument

            # this print does not work anymore as the context has been removed more or less
            # if i % 1000 == 0:
            #     print("Epoch:", epoch, i, "/", iterations, "{0:.2f}%".format((i * 100) / iterations))
            #     print(time.time() - start_time)
            #     print_info(i, context, target_word, outputted_word, loss)

            # # Clip gradients to avoid gradient explosion
            # torch.nn.utils.clip_grad_norm(ran.parameters(), LOSS_CLIP)
            for p in ran.parameters():
                if p.requires_grad:
                    p.data.add_(-learning_rate, p.grad.data)

            total_loss += loss.data

            if batch % BATCH_LOG_INTERVAL == 0 and batch > 0:
                cur_loss = total_loss[0] / BATCH_LOG_INTERVAL
                print("current perplexity:", exp(cur_loss))
                total_loss = 0

        save_model(ran, epoch)

train_RAN(training_data=training_data, \
          learning_rate=LEARNING_RATE, \
          epochs=EPOCHS, \
          vocab_size=vocab_size, \
          word_embeddings=word_embeddings, \
          use_GPU=use_GPU)
