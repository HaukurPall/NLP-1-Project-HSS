test_data = DataReader("data/penn/test.txt", read_limit=READ_LIMIT)
# TODO: Load model from file
# evaluate_model(trained_model, test_data, word_to_index)

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
    probability = get_max_value_and_index(log_probs)[1]
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
