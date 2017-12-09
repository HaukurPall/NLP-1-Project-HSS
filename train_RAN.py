import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from RAN import RAN

from data_reader import DataReader, get_pretrained_word_indexes
from data_reader import update_word_indexes_vocab, get_embeddings_matrix
from ngram_helper import extract_list_of_ngrams

from math import inf
from collections import defaultdict
import time

NGRAM_SIZE = 5
HIDDEN_SIZE = 128
CONTEXT_SIZE = NGRAM_SIZE - 1
WORD_EMBEDDINGS_DIMENSION = 50
LEARNING_RATE = 0.005
LOSS_CLIP = 30
READ_LIMIT = inf

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

def save_model(model):
    torch.save(model.state_dict(), "saved_model.pt")

def context_to_variable(context):
    context_tensor = torch.FloatTensor(CONTEXT_SIZE, WORD_EMBEDDINGS_DIMENSION)
    for i in range(len(context)):
        context_tensor[i] = word_embeddings[word_to_index[context[i]]]

    return Variable(context_tensor)

def word_to_variable(word): # Might not be necessary
    word_tensor = word_embeddings[word_to_index[word]]
    return Variable(word_tensor)

def word_from_output(output):
    top_n, top_i = output.data.topk(1)
    word_index = top_i[0][0]

    return index_to_word[word_index], word_index

def train_RAN():
    criterion = nn.NLLLoss()

    ran = RAN(WORD_EMBEDDINGS_DIMENSION, HIDDEN_SIZE, vocab_size)
    hidden = ran.init_hidden()
    ran.zero_grad()

    iterations = len(ngrams)

    for i in range(iterations):
        context, target_word = ngrams[i]
        context_variable = context_to_variable(context)

        for i in range(context_variable.size()[0]):
            output, hidden = ran(context_variable[i].view(1, -1), hidden)

        target_index = word_to_index[target_word]
        target_variable = Variable(torch.LongTensor([target_index]))
        outputed_word = word_from_output(output)[0]

        loss = criterion(output, target_variable)
        # loss.data.clamp_(-LOSS_CLIP, LOSS_CLIP)
        print(i, "Context: ", context, "Target word was: ", target_word)
        print("Predicted word was: ", outputed_word)
        print("Loss: ", loss)
        if loss.data[0] == "nan":
            print("Gradient exploded...")
            break
        if target_word == outputed_word[0]:
            print("####### Prediction was right!! ", context, target_word)
        loss.backward(retain_graph=True) # Don't understand why we need this argument

        for p in ran.parameters():
            p.data.add_(-LEARNING_RATE, p.grad.data)
train_RAN()
