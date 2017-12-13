# NLP-1 Language modeling project
A project to test different language models.
1. KenLM
2. FFNN
3. RAN

# Authors
Haukur Páll Jónsson, Santhosh Kumar Rajamanickam, Stian Steinbakken

# KenLM
A standard N-gram, probabilistic model which uses Kneser–Ney smoothing.
https://kheafield.com/code/kenlm/estimation/
http://masatohagiwara.net/training-an-n-gram-language-model-and-estimating-sentence-probability.html

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
  cd ../..

The result of the compilation is committed in the repo but might need recompilation in order to run.

## Running
We "ignore" UNK words instead of treating them as a singluar word.

  kenlm/build/bin/lmplz --skip_symbols -S 40% -o 3 < data/penn/train.txt > lm/penn.arpa
  kenlm/build/bin/build_binary lm/penn.arpa lm/penn.klm

Querying the model

  python3.6 ken.py lm/penn.klm

or

  cat data/penn/test.txt | kenlm/build/bin/query lm/penn.klm

## Results
As expected we get higher perplexity by ignoring the <unk> instead of treating all of them as the same word.

N = 3

  Perplexity including OOVs:	278.1419220649369
  Perplexity excluding OOVs:	169.1728707928899
  OOVs:	4794
  Tokens:	82430
  Name:query	VmPeak:40216 kB	VmRSS:4144 kB	RSSMax:20928 kB	user:0.036	sys:0.096	CPU:0.133185	real:3.53189

N = 4

  Perplexity including OOVs:	268.04674615356913
  Perplexity excluding OOVs:	162.85524251588026
  OOVs:	4794
  Tokens:	82430
  Name:query	VmPeak:55648 kB	VmRSS:4216 kB	RSSMax:36576 kB	user:0.044	sys:0.092	CPU:0.138543	real:3.52924

N = 5

  Perplexity including OOVs:	265.48538591471396
  Perplexity excluding OOVs:	161.2635989567618
  OOVs:	4794
  Tokens:	82430
  Name:query	VmPeak:71900 kB	VmRSS:4304 kB	RSSMax:52892 kB	user:0.04	sys:0.108	CPU:0.151068	real:3.53441

# Neural Networks

## RAN
Original code using tensorflow https://github.com/kentonl/ran
Pytorch: https://github.com/bheinzerling/ran

# Data
- Training/Test/Valid = Penn
- Embeddings = https://nlp.stanford.edu/data/glove.6B.zip

# Notes
use dropout (see https://arxiv.org/abs/1510.00726 section 6.4 ) - this randomly drops out e.g. 50% of the units of your layer during training, making sure the network cannot rely on a single specific weight at any time. In Pytorch, dropout is enabled if you run ```model.train()``` in your training loop, and it is disabled if you run ```model.eval()```. You can just specify it as another layer in your model:
self.dropout = nn.Dropout(p=0.5, inplace=False)
use weight decay. This is implemented in optim.SGD as a parameter. You can use e.g. weight_decay=1e-6. This prevents weights from growing too large, which helps generalization. Weight decay is similar to L2 regularization (see https://algorithmsdatascience.quora.com/Weight-decay-vs-L2-regularization )
and (3), hopefully you are doing this already: shuffle your training set during training! an easy way to do this is to select a random integer in the range of the number of tokens in your training set, and then get the (n-1) words after that to make a training example. If you do this many times, on expectation you will cover the whole training set after seeing M examples (where M is the number of tokens in train).

From https://openreview.net/forum?id=HJOQ7MgAW&noteId=ByUL_Swbz:
All comparison against LSTMs (and of course, any other architectures) should be done carefully, especially when there is evidence that LSTMs weren't properly tuned. For instance, Melis et al. (https://arxiv.org/pdf/1707.05589.pdf) point out that LSTMs if tuned properly can outperform most state-of-the-art models. In Melis et al. LSTMs achieve 59.6 on PTB, whereas the authors' LSTMs reach 78.8, which is much much far behind.
