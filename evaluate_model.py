from NNLM import NGramLanguageModeler
from data_reader import DataReader, get_pretrained_word_indexes
from data_reader import update_word_indexes_vocab, get_embeddings_matrix
from ngram_helper import extract_list_of_ngrams

from train_model_penn_tree import get_max_index_and_value

import torch
import torch.autograd as autograd

from math import inf

READ_LIMIT = inf
WORD_EMBEDDINGS_DIMENSION = 50
NGRAM_SIZE = 3
CONTEXT_SIZE = NGRAM_SIZE - 1

pretrained_filepath = "data/glove.6B.50d.txt"
training_data_filepath = "data/train.txt"
test_data_filepath = "data/test.txt"

def get_target_else_unknown(lookuptable, target):
  if target in lookuptable:
    return lookuptable[target]

  else:
    return lookuptable["unknown"]

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
    probability = get_max_index_and_value(log_probs)[1]
    probability_of_sentence += probability

  return perplexity(probability_of_sentence, len(sentence))

def calculate_average_perplexity(sentences, model, word_to_index):
  total_perplexity = 0

  for sentence in sentences:
    total_perplexity += calculate_perplexity(sentence, model, word_to_index)

  return total_perplexity / len(sentences)

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

    predicted_word_index = get_max_index_and_value(log_probs)[0]
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
  # First load the models as it was, that is loading it with the training sizes and vocab
  training_data = DataReader(training_data_filepath)

  vocab = training_data.vocab

  # Build a list of trigrams
  words = training_data.get_words()

  # Get the pretrained word vectors
  word_to_index, embed_dict = get_pretrained_word_indexes(pretrained_filepath)

  # Update word_to_index and vocabulary
  word_to_index, vocab = update_word_indexes_vocab(word_to_index, vocab)

  # Get the numpy matrix containing the pretrained word vectors
  # with randomly initialized unknown words from the corpus
  word_embeddings = get_embeddings_matrix(word_to_index, embed_dict, WORD_EMBEDDINGS_DIMENSION)

  model = NGramLanguageModeler(len(vocab), 50, CONTEXT_SIZE, word_embeddings)
  model.load_state_dict(torch.load("AWS_model.pt"))

  test_data = DataReader(test_data_filepath, read_limit=READ_LIMIT)

  evaluate_model(model, test_data, word_to_index)

main()
