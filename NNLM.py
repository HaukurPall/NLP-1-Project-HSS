import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_reader import DataReader
from ngram_helper import extract_list_of_ngrams

# Constants
WORD_EMBEDDINGS_DIMENSION = 50
NGRAM_SIZE = 3
CONTEXT_SIZE = NGRAM_SIZE - 1
LEARNING_RATE = 0.001
NUMBER_OF_TRAINING_EPOCHS = 100

training_data = DataReader("data/penn/train.txt", read_limit=10)

word_to_index = training_data.get_word_indexes()
vocab_size = len(word_to_index)

words = training_data.get_words()
trigrams = extract_list_of_ngrams(words, NGRAM_SIZE)

class NGramLanguageModeler(nn.Module):

  # Taken straight out of the example. We might want to make our own adjustments here

  def __init__(self, vocab_size, embedding_dim, context_size):
    super(NGramLanguageModeler, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.linear1 = nn.Linear(context_size * embedding_dim, 128)
    self.linear2 = nn.Linear(128, vocab_size)

  def forward(self, inputs):
    embeds = self.embeddings(inputs).view((1, -1))
    out = F.relu(self.linear1(embeds))
    out = self.linear2(out)
    log_probs = F.log_softmax(out)
    return log_probs

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(vocab_size, WORD_EMBEDDINGS_DIMENSION, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUMBER_OF_TRAINING_EPOCHS):
  print("Epoch:", epoch)
  total_loss = torch.Tensor([0])
  for context, target in trigrams:

    # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
    # into integer indices and wrap them in variables)
    context_indexes = [word_to_index[w] for w in context]
    context_var = autograd.Variable(torch.LongTensor(context_indexes))

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
        torch.LongTensor([word_to_index[target]])))

    # Step 5. Do the backward pass and update the gradient
    loss.backward()
    optimizer.step()

    total_loss += loss.data
  losses.append(total_loss)
print(losses)
