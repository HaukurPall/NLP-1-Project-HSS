import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_reader import get_pretrained_embeddings
import time

# Constants
WORD_EMBEDDINGS_DIMENSION = 50
LEARNING_RATE = 0.001
NUMBER_OF_TRAINING_EPOCHS = 10
LOSS_THRESHOLD = 0.01

class NGramLanguageModeler(nn.Module):

  # Taken straight out of the example. We might want to make our own adjustments here
  def __init__(self, vocab_size, embedding_dim, context_size, word_embeddings):
    super(NGramLanguageModeler, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.embeddings.weight.data.copy_(word_embeddings)
    self.embeddings.weight.requires_grad = False # Do not train the pre-calculated embeddings
    self.embeddings = self.embeddings.cuda()
    self.layer1 = nn.Linear(context_size * embedding_dim, vocab_size)
    self.layer1 = self.layer1.cuda()

  def forward(self, inputs):
    embeds = self.embeddings(inputs).view((1, -1))
    out = F.tanh(self.layer1(embeds))
    log_probs = F.log_softmax(out)
    return log_probs

def get_target_else_unknown(lookuptable, target):
  # TODO: Handle cyclic import so we can import this from train_and_evaluate
  # instead of duplicate code
  if target in lookuptable:
    return lookuptable[target]
  else:
    return lookuptable["unknown"]

def print_info(i, start_time, trigrams):
  percentage_done = i/ len(trigrams) * 100
  t = time.time() - start_time

  print("\n##############################\n")
  print(str(i) + "/" + str(len(trigrams)), "Epoch is ", "{0:.0f}%".format(percentage_done), "done")
  print("Current runtime", t, "seconds")
  minutes_left_of_epoch = (t * (100 / percentage_done) / 60)
  print("Estimated minutes left of current epoch:", minutes_left_of_epoch)
  hours, minutes = divmod(minutes_left_of_epoch, 60)
  print("Which is: ", "%02d:%02d"%(hours,minutes))
def train_model(trigrams, vocab_size, CONTEXT_SIZE, word_to_index, word_embeddings):

  start_time = time.time()

  total_losses = []
  loss_function = nn.NLLLoss().cuda()
  model = NGramLanguageModeler(vocab_size, WORD_EMBEDDINGS_DIMENSION, CONTEXT_SIZE, word_embeddings)
  optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

  for epoch in range(NUMBER_OF_TRAINING_EPOCHS):
    print("epoch:", epoch)
    total_loss = torch.Tensor([0]).cuda()
    for i, (context, target) in enumerate(trigrams):
      if i % 1000 == 1:
        print_info(i, start_time, trigrams)

      # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
      # into integer indices and wrap them in variables)
      context_indexes = [get_target_else_unknown(word_to_index, w) for w in context]
      context_var = autograd.Variable(torch.LongTensor(context_indexes).cuda())

      # Step 2. Recall that torch *accumulates* gradients. Before passing in a
      # new instance, you need to zero out the gradients from the old
      # instance
      model.zero_grad()

      # Step 3. Run the forward pass, getting log probabilities over next
      # words
      log_probs = model(context_var)

      # Step 4. Compute your loss function. (Again, Torch wants the target
      # word wrapped in a variable)
      target_index = get_target_else_unknown(word_to_index, target)

      loss = loss_function(log_probs, autograd.Variable(
          torch.LongTensor([target_index]).cuda()))

      # Step 5. Do the backward pass and update the gradient
      loss.backward()
      optimizer.step()

      total_loss += loss.data

    total_losses.append(total_loss)
    if len(total_losses) > 1:
      l = abs((total_losses[-1] - total_losses[-2])[0])
      print("Current difference in loss:", l)
      if l < LOSS_THRESHOLD:
        break

  return model.cuda()
