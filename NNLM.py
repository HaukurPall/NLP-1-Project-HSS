import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_reader import DataReader, get_pretrained_word_indexes
from data_reader import update_word_indexes_vocab, get_embeddings_matrix
from ngram_helper import extract_list_of_ngrams
from math import inf
import time

# Constants
READ_LIMIT = inf # Manually reset if we want faster processing
CONTEXT_SIZE = 2 # Trigram
EMBEDDING_DIM = 50 # No. of embedding dimensions of GLOVE
LEARNING_RATE = 0.001
NO_OF_EPOCHS = 2

# File paths
pretrained_filepath = "data/glove.6B.50d.txt"
training_data_filepath = "data/train.txt"

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, word_embeddings):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.data.copy_(torch.from_numpy(word_embeddings))
        self.linear1 = nn.Linear(context_size * embedding_dim, vocab_size)
        self.embeddings.weight.requires_grad = False # Do not train the pre-calculated embeddings

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.tanh(self.linear1(embeds))
        log_probs = F.log_softmax(out)
        return log_probs

# Read corpus and compile the vocabulary
training_data = DataReader(training_data_filepath, read_limit=READ_LIMIT)
vocab = training_data.vocab
# Build a list of trigrams
words = training_data.get_words()
trigrams = extract_list_of_ngrams(words, CONTEXT_SIZE + 1)
# Get the pretrained word vectors
word_to_ix, embed_dict = get_pretrained_word_indexes(pretrained_filepath)
# Update word_to_ix and vocabulary
word_to_ix, vocab = update_word_indexes_vocab(word_to_ix, vocab)
# Get the numpy matrix containing the pretrained word vectors
# with randomly initialized unknown words from the corpus
word_embeddings = get_embeddings_matrix(word_to_ix, embed_dict, EMBEDDING_DIM)

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, word_embeddings)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

epoch_times = []

for epoch in range(NO_OF_EPOCHS):

    start_time = time.time()

    print("Epoch: {}".format(epoch))
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_var)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    losses.append(total_loss)

    t = time.time() - start_time
    epoch_times.append(t)
print(losses)  # The loss decreased every iteration over the training data!
print(epoch_times)
