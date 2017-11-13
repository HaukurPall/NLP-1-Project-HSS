from collections import defaultdict
from math import inf

class DataReader:
  def __init__(self, filepath, read_limit=inf):
    self.words = []
    with open(filepath, "r") as f:
      for line_number, line in enumerate(f.readlines()):
        for word in line.split():
          self.words.append(word)
        if line_number > read_limit: break


  def get_words(self):
    return self.words

  def get_word_indexes(self):
    # Returns a dictionary containing all the words in the vocabulary
    # and their respective indexes.
    word_to_index = defaultdict(lambda: len(word_to_index))
    for word in self.words:
      word_to_index[word]

    return word_to_index
