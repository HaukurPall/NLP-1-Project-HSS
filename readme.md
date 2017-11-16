# NLP-1 Language modeling project
A project to test different language models.
1. KenLM
2. FFNN
3. ??

# Authors
Haukur Páll Jónsson, Santhosh Kumar Rajamanickam, Stian Steinbakken

# KenLM
A standard N-gram, probabilistic model which uses Kneser–Ney smoothing.
https://kheafield.com/code/kenlm/estimation/

## Setup
- We cloned the kenlm project to our project and build it from source
- First install boost_1.6:

  apt-get install python3.6-gdbm python3.6-dev
  http://victor.chahuneau.fr/notes/2012/07/03/kenlm.html
  sudo apt-get install libbz2-dev
  # Download boost 1.6 tar.gz
  tar -xvf boost_1_60_0.tar.gz
  cd boost_1_60_0
  ./bootstrap.sh
  ./b2 -j 4
  ./b2 install # might need sudo
  popd

Create the KenLM binaries. We need to build the whole thing as we have <unk> in our vocabulary and we need to tell KenLM to treat these words as whitespaces. KenLM handles <unk> for us and gives an error if provided in the corpus.

  wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
  mkdir kenlm/build
  cd kenlm/build
  cmake ..
  make -j2

Running it

  kenlm/build/bin/lmplz --skip_symbols -S 40% -o 3 < data/penn/train.txt > lm/penn.arpa
  kenlm/build/bin/build_binary lm/penn.arpa lm/penn.klm

Querying the model

  python3.6 ken.py lm/penn.klm 


# Neural Networks



# Data
- Training/Test/Valid = Penn
- Embeddings = https://nlp.stanford.edu/data/glove.6B.zip
