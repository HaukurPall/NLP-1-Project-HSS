from NNLM import train_model
from data_reader import DataReader, get_pretrained_word_indexes
from data_reader import update_word_indexes_vocab, get_embeddings_matrix
from ngram_helper import extract_list_of_ngrams
from math import inf, pow, log

import torch
import torch.autograd as autograd

import time

# Constants
READ_LIMIT = 1
NGRAM_SIZE = 3
CONTEXT_SIZE = NGRAM_SIZE - 1
WORD_EMBEDDINGS_DIMENSION = 50

pretrained_filepath = "data/glove.6B.50d.txt"
training_data_filepath = "data/train.txt"

def get_max_value_and_index(tensor):
  # this is probably not necessary,
  # but I can't figure out how to get the index of the max element of a tensor.
  best_value = -inf
  best_index = -1
  for i, value in enumerate(tensor.data[0]):
    if value > best_value:
      best_value = value
      best_index = i

  return best_index, best_value

def calculate_perplexity(sentence, model, word_to_index):

  def perplexity(log_probability, n):
    if 10 ** log_probability == 0:
      return inf # TODO: Figure out how to deal with very low probabilities
    return pow(10 ** log_probability, -(1/n))

  sentence = sentence.split()
  trigrams = extract_list_of_ngrams(sentence, NGRAM_SIZE)
  probability_of_sentence = 0

  for context, target in trigrams:
    context_indexes = [get_target_else_unknown(word_to_index, w) for w in context]
    context_var = autograd.Variable(torch.LongTensor(context_indexes).cuda(async=True))

    log_probs = model(context_var)
    probability = get_max_value_and_index(log_probs)[1]
    probability_of_sentence += probability

  return perplexity(probability_of_sentence, len(sentence))

def calculate_average_perplexity(sentences, model, word_to_index):
  total_perplexity = 0

  for sentence in sentences:
    total_perplexity += calculate_perplexity(sentence, model, word_to_index)

  return total_perplexity / len(sentences)

def get_target_else_unknown(lookuptable, target):
  if target in lookuptable:
    return lookuptable[target]

  else:
    return lookuptable["unknown"]

def save_model(model):
  torch.save(model.state_dict(), "saved_model.pt")

def evaluate_model(model, test_data, word_to_index):
  test_words = test_data.get_words()

  test_trigrams = extract_list_of_ngrams(test_words, NGRAM_SIZE)

  correct_predictions = 0
  total_predictions = 0

  for i, (context, target) in enumerate(test_trigrams):
    print("Testing trigram nr", i, "out of", len(test_trigrams), "trigrams")
    context_indexes = [get_target_else_unknown(word_to_index, w) for w in context]

    context_var = autograd.Variable(torch.LongTensor(context_indexes).cuda(async=True))

    log_probs = model(context_var)

    predicted_word_index = get_max_value_and_index(log_probs)[0]
    assert predicted_word_index != -1, "The index of the predicted word was -1"

    print("Predicted:", predicted_word_index, "Actual:", get_target_else_unknown(word_to_index, target))
    if predicted_word_index == get_target_else_unknown(word_to_index, target):
      correct_predictions += 1

    total_predictions += 1

  print("Correct predictions:", correct_predictions/total_predictions)
  print("Correct predictions:", correct_predictions, "out of", total_predictions)

  print("####### Calculating perplexities #######")
  average_perplexity = calculate_average_perplexity(test_data.get_sentences(), model, word_to_index)
  print("Average perplexity:", average_perplexity)

def main():
  training_data = DataReader("data/penn/train.txt", read_limit=READ_LIMIT)

  # Read corpus and compile the vocabulary
  training_data = DataReader(training_data_filepath, read_limit=READ_LIMIT)
  vocab = training_data.vocab
  # Build a list of trigrams
  words = training_data.get_words()
  trigrams = extract_list_of_ngrams(words, NGRAM_SIZE)
  # Get the pretrained word vectors
  word_to_index, embed_dict = get_pretrained_word_indexes(pretrained_filepath)
  # Update word_to_index and vocabulary
  word_to_index, vocab = update_word_indexes_vocab(word_to_index, vocab)
  # Get the numpy matrix containing the pretrained word vectors
  # with randomly initialized unknown words from the corpus
  word_embeddings = get_embeddings_matrix(word_to_index, embed_dict, WORD_EMBEDDINGS_DIMENSION)

  print("Done reading data and creating trigrams")

  start_time = time.time()

  trained_model = train_model(trigrams, len(vocab), CONTEXT_SIZE, word_to_index, \
      word_embeddings, WORD_EMBEDDINGS_DIMENSION)
  save_model(trained_model)

  print("Final training time: ", time.time() - start_time)

  test_data = DataReader("data/penn/test.txt", read_limit=READ_LIMIT)
  evaluate_model(trained_model, test_data, word_to_index)

main()
