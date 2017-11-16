def extract_list_of_ngrams(words, n):
  ngrams = []

  for i in range(len(words) - n):
    history = [words[j] for j in range(i, i + n - 1)]

    word = words[i + n - 1]
    ngrams.append((history, word))

  return ngrams
