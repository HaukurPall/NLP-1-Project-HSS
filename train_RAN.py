import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from RAN import RAN

from data_reader import DataReader, get_pretrained_word_indexes
from data_reader import update_word_indexes_vocab, get_embeddings_matrix
from ngram_helper import extract_list_of_ngrams

from math import inf
from collections import defaultdict
import time

#### Options

use_GPU = True

#### Constants

NGRAM_SIZE = 5
BATCH_SIZE = 20
CONTEXT_SIZE = NGRAM_SIZE - 1
WORD_EMBEDDINGS_DIMENSION = 50
LOSS_CLIP = 30
READ_LIMIT = inf

learning_rate = 0.01
pretrained_embeddings_filepath = "data/glove.6B.50d.txt"
training_data_filepath = "data/train.txt"

###### Read in all the data outside of functions so we can refer to them 'globally' #####
# Read corpus and compile the vocabulary
training_data = DataReader(training_data_filepath, read_limit=READ_LIMIT)
vocab = training_data.get_vocabulary()

# Build a list of ngrams
words = training_data.get_words()
ngrams = extract_list_of_ngrams(words, NGRAM_SIZE)

# Get the pretrained word vectors and build a word_to_index based on it
word_to_index, embed_dict = get_pretrained_word_indexes(pretrained_embeddings_filepath)

# Update word_to_index and vocabulary
word_to_index, vocab = update_word_indexes_vocab(word_to_index, vocab)
vocab_size = len(vocab)

word_to_index = defaultdict(lambda: "unknown", word_to_index)
index_to_word = {i : word for word, i in word_to_index.items()}

# Get the numpy matrix containing the pretrained word vectors
# with randomly initialized unknown words for words that do not occur in the pretrained
# embeddings, but that occur in the corpus (roughly 73 words)
word_embeddings = get_embeddings_matrix(word_to_index, embed_dict, WORD_EMBEDDINGS_DIMENSION)

# Convert the numpy matrix to a torch tensor
word_embeddings = torch.from_numpy(word_embeddings)

print("### Done reading data and creating ngrams ###")

def save_model(model, epoch):
    torch.save(model.state_dict(), str(epoch) + "_saved_model.pt")

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

def get_batch(i):
    context_batch = tensor.LongTensor(BATCH_SIZE, CONTEXT_SIZE)
    target_batch = []
    for x in range(i, i + BATCH_SIZE):
        context_batch.append(ngrams[x][0])
        target_batch.append(ngrams[x][1])

def train_RAN(epochs):
    criterion = nn.NLLLoss() if not use_GPU else nn.NLLLoss().cuda()

    ran = RAN(WORD_EMBEDDINGS_DIMENSION * CONTEXT_SIZE, vocab_size, word_embeddings, use_GPU)

    if use_GPU:
        ran = ran.cuda()

    iterations = len(ngrams)
    start_time = time.time()

    for epoch in range(epochs):
        # learning_rate *= 0.95 # Reduce learning rate each epoch

        for i in range(iterations - BATCH_SIZE):
            ran.zero_grad()
            hidden = ran.init_hidden()

            context, target_word = ngrams[i]
            context_variable = context_to_variable(context)

            # TODO: batch
            output, hidden = ran(context_variable, hidden)

            target_index = word_to_index[target_word]
            target_variable = Variable(torch.LongTensor([target_index]))

            if use_GPU:
                target_variable = target_variable.cuda()

            outputted_word = word_from_output(output)[0]

            # print(output, target_variable)
            loss = criterion(output, target_variable)

            # # Clip gradients to avoid gradient explosion
            # torch.nn.utils.clip_grad_norm(ran.parameters(), LOSS_CLIP)

            loss.backward(retain_graph=True) # Don't understand why we need this argument

            if i % 1000 == 0:
                print("Epoch:", epoch, i, "/", iterations, "{0:.2f}%".format((i * 100) / iterations))
                print(time.time() - start_time)
                print_info(i, context, target_word, outputted_word, loss)

            for p in ran.parameters():
                p.data.add_(-learning_rate, p.grad.data)

        save_model(ran, epoch)

train_RAN(10)
