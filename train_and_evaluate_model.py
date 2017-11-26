from NNLM import train_model
from data_reader import DataReader, get_pretrained_word_indexes, get_pretrained_embeddings
from ngram_helper import extract_list_of_ngrams
from math import inf, pow, log

import torch
import torch.autograd as autograd

import time

# Constants
READ_LIMIT = inf
NGRAM_SIZE = 3
CONTEXT_SIZE = NGRAM_SIZE - 1

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

def calculate_perplexity(sentence, model):

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

def calculate_average_perplexity(sentences, model):
  total_perplexity = 0

  for sentence in sentences:
    total_perplexity += calculate_perplexity(sentence, model)

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

  for context, target in test_trigrams:
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
  average_perplexity = calculate_average_perplexity(training_data.get_sentences(), model)
  print("Average perplexity:", average_perplexity)

def main():
  training_data = DataReader("data/penn/train.txt", read_limit=READ_LIMIT)

  predefined_vocabulary, word_embeddings = get_pretrained_embeddings()
  vocab_size = len(predefined_vocabulary)
  word_to_index = get_pretrained_word_indexes()

  words = training_data.get_words()
  trigrams = extract_list_of_ngrams(words, NGRAM_SIZE)
  print("Done reading data and creating trigrams")

  start_time = time.time()

  trained_model = train_model(trigrams, vocab_size, CONTEXT_SIZE, word_to_index, word_embeddings)
  save_model(trained_model)

  print("Final training time: ", time.time() - start_time)

  # test_data = DataReader("data/penn/test.txt", read_limit=READ_LIMIT)
  # evaluate_model(trained_model, test_data, word_to_index)

main()
