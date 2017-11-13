from collections import defaultdict

class DataReader:
  def __init__(self, filepath):
    self.words = []
    with open(filepath, "r") as f:
      for line in f.readlines():
        for word in line.split():
          self.words.append(word)


  def get_word_indexes(self):
    # Returns a dictionary containing all the words in the vocabulary
    # and their respective indexes.
    word_to_index = defaultdict(lambda: len(word_to_index))
    for word in self.words:
      word_to_index[word]

    return word_to_index
