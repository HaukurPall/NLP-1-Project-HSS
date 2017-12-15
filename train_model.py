import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from RAN import RAN
from LSTM import LSTM

from data_reader import DataReader
from ngram_helper import extract_list_of_ngrams
import utilities

from math import inf, exp
from collections import defaultdict
import time
import sys
import argparse

#### Options


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--use_pretrained', action='store_true')

args = parser.parse_args()

use_GPU = args.cuda
use_pretrained = args.use_pretrained
print("Using GPU" if use_GPU else "Running without GPU")
model = args.model

#### Constants
BATCH_LOG_INTERVAL = 100
READ_LIMIT = inf

EPOCHS = 100
BATCH_SIZE = 64
EVAL_BATCH_SIZE = BATCH_SIZE
CONTEXT_SIZE = 35
WORD_EMBEDDINGS_DIMENSION = 300
LEARNING_RATE = 5
MIN_LEARNING_RATE = 0.001
LOSS_CLIP = 10
NUM_HIDDEN_UNITS = WORD_EMBEDDINGS_DIMENSION
NUM_LAYERS = 1
DROPOUT_PROB = 0.5

training_data_filepath = "data/train.txt"
validation_data_filepath = "data/valid.txt"
test_data_filepath = "data/test.txt"

perplexity_filepath, model_filepath = utilities.create_filepaths(model=model, \
                                                     batch_size=LEARNING_RATE, \
                                                     emb_dim=WORD_EMBEDDINGS_DIMENSION, \
                                                     lr=LEARNING_RATE)


training_data = DataReader(training_data_filepath, read_limit=READ_LIMIT)
validation_data = DataReader(validation_data_filepath, read_limit=READ_LIMIT)
test_data = DataReader(test_data_filepath, read_limit=READ_LIMIT)

vocab_size, word_to_index, index_to_word, word_embeddings = utilities.prepare_dictionaries(training_data=training_data, \
                                                                                 emb_dim=WORD_EMBEDDINGS_DIMENSION, \
                                                                                 use_pretrained=use_pretrained)

###### Read in all the data outside of functions so we can refer to them 'globally' #####
# Read corpus and compile the vocabulary

training_data = utilities.get_word_tensors(data=training_data, word_to_index=word_to_index)
# When we have unseen words in the validation set we default to unknown
word_to_index = defaultdict(lambda: word_to_index["unknown"], word_to_index)

###### Same operations for the validation and test dataset ######
validation_data = utilities.get_word_tensors(data=validation_data, word_to_index=word_to_index)
test_data = utilities.get_word_tensors(data=test_data, word_to_index=word_to_index)

print("### Done reading data ###")

# We change the dataset from a one dimensional vector to a x*batch_size matrix
training_data = utilities.batchify(training_data, BATCH_SIZE, use_GPU=use_GPU)
validation_data = utilities.batchify(validation_data, BATCH_SIZE, use_GPU=use_GPU)
test_data = utilities.batchify(test_data, BATCH_SIZE, use_GPU=use_GPU)

def save_model(model, epoch, model_filepath):
    torch.save(model.state_dict(), model_filepath + "_" + str(epoch))

def evaluate(data_source, model, criterion, use_GPU):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(EVAL_BATCH_SIZE)

    for i in range(0, data_source.size(0) - 1, CONTEXT_SIZE):
        data, targets = utilities.get_batch(data_source, CONTEXT_SIZE, i, use_GPU, evaluation=True)

        output, hidden = model(data, hidden)
        output_flat = output.view(-1, vocab_size)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)

    model.train()

    return total_loss[0] / len(data_source)

def save_perplexity(filepath, perplexity, optimization_steps):
    with open(filepath, "a") as f:
        f.write(str(optimization_steps) + " " + str(perplexity) + "\n")

    return True

def repackage_hidden(hidden):
    if type(hidden) == Variable:
        return Variable(hidden.data)
    else:
        return tuple(repackage_hidden(v) for v in hidden)

def has_improved(checkpoint_perplexities, prev_best):
    if len(checkpoint_perplexities) < 30:
        # We don't want to reduce the learning rate if have not made 30 checkpoints yet
        return True, prev_best

    improved = False

    best = inf
    for perp in checkpoint_perplexities[-30:]:
        if perp < best:
            best = perp

    if best < prev_best:
        improved = True

    return improved, best
    # return checkpoint_perplexities[-30] - checkpoint_perplexities[-1] > IMPROVEMENT_EPSILON

def main(model, \
                training_data, \
                validation_data, \
                test_data, \
                learning_rate, \
                epochs, \
                vocab_size, \
                use_GPU):

    criterion = nn.CrossEntropyLoss() if not use_GPU else nn.CrossEntropyLoss().cuda()

    start_time = time.time()

    optimization_steps = 0
    checkpoint_perplexities = []
    best_prev = inf

    for epoch in range(epochs):
        # turn on dropouts
        total_loss = 0
        model.train()

        hidden = model.init_hidden(BATCH_SIZE)

        for batch, i in enumerate(range(0, training_data.size(0) - 1, CONTEXT_SIZE)):
            hidden = repackage_hidden(hidden)
            model.zero_grad()

            data, targets = utilities.get_batch(training_data, CONTEXT_SIZE, i, use_GPU)

            output, hidden = model(data, hidden)

            loss = criterion(output.view(-1, vocab_size), targets)

            loss.backward()

            optimization_steps += 1
            if optimization_steps % 100 == 0:

                checkpoint_perplexities.append(exp(evaluate(validation_data, model, criterion, use_GPU)))

                if optimization_steps % 3000 == 0:

                    improved, best_prev = has_improved(checkpoint_perplexities, best_prev)
                    if not improved:
                        learning_rate = max(learning_rate*0.1, MIN_LEARNING_RATE)
                        print("Reduced learning rate to", learning_rate)

            # Clip gradients to avoid gradient explosion
            torch.nn.utils.clip_grad_norm(model.parameters(), LOSS_CLIP)

            for p in model.parameters():
                if p.requires_grad:
                    p.data.add_(-learning_rate, p.grad.data)

            total_loss += loss.data

            if batch % BATCH_LOG_INTERVAL == 0 and batch > 0:
                cur_loss = total_loss[0] / BATCH_LOG_INTERVAL
                total_loss = 0

        save_model(model, epoch, model_filepath)
        average_loss = evaluate(validation_data, model, criterion, use_GPU)
        validation_perplexity = exp(average_loss)

        print("\nValidation perplexity", validation_perplexity, "Epoch", epoch, "\n")
        save_perplexity(perplexity_filepath, validation_perplexity, optimization_steps)

    test_loss = evaluate(test_data, model, criterion, use_GPU)
    test_perplexity = exp(test_loss)
    print("############## FINAL TEST PERPLEXITY ################")
    print(test_perplexity)
    save_perplexity(perplexity_filepath, test_perplexity, "FINAL")

if model == "RAN":
    model = RAN(vocab_size, \
                WORD_EMBEDDINGS_DIMENSION, \
                NUM_LAYERS, \
                DROPOUT_PROB, \
                word_embeddings, \
                use_pretrained=use_pretrained, \
                use_GPU=use_GPU)
elif model == "LSTM":
    model = LSTM(vocab_size, \
                 WORD_EMBEDDINGS_DIMENSION, \
                 NUM_LAYERS, \
                 DROPOUT_PROB, \
                 word_embeddings, \
                 use_pretrained=use_pretrained, \
                 use_GPU=use_GPU)

if use_GPU:
    model.cuda()
main(model=model, \
      training_data=training_data, \
      validation_data=validation_data, \
      test_data=test_data, \
      learning_rate=LEARNING_RATE, \
      epochs=EPOCHS, \
      vocab_size=vocab_size, \
      use_GPU=use_GPU)
