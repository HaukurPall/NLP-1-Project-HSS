import torch
from torch.autograd import Variable
from datetime import datetime
from data_reader import update_word_indexes_vocab, get_embeddings_matrix, get_pretrained_word_indexes

def create_filepaths(model, batch_size, emb_dim, lr):
    timestamp = str(datetime.now()).split()[1][:8].replace(":", "_")
    timestamp_signature = "{}_{}_batch_{:d}_embed_{}_learn_{}".format(model, \
                                                                      timestamp, \
                                                                      batch_size, \
                                                                      emb_dim, \
                                                                      str(lr)[:4])
    perplexity_filepath = "perplexities/first_offical_run_" + timestamp_signature + ".txt"
    model_filepath = "saved_models/{}_{}.pt".format(model, timestamp_signature + str(epoch))
    return perplexity_filepath, model_filepath


def prepare_dictionaries(training_data, emb_dim, use_pretrained=True):
    vocab = training_data.get_vocabulary()
    word_to_index, index_to_word = training_data.get_word_to_index_to_word()
    word_embeddings = None

    if use_pretrained:
        pretrained_embeddings_filepath = "data/glove.6B.{}d.txt".format(emb_dim)
        # Get the pretrained word vectors and build a word_to_index based on it
        _, embed_dict = get_pretrained_word_indexes(pretrained_embeddings_filepath)

        # Get the numpy matrix containing the pretrained word vectors
        # with randomly initialized unknown words for words that do not occur in the pretrained
        # embeddings, but that occur in the corpus (roughly 73 words)
        word_embeddings = get_embeddings_matrix(word_to_index, embed_dict, emb_dim)

        # Convert the numpy matrix to a torch tensor
        word_embeddings = torch.from_numpy(word_embeddings)
    return len(vocab), word_to_index, index_to_word, word_embeddings

def get_word_tensors(data, word_to_index):
    words = data.get_words()
    # now the training data is a one dimensional vector of indexes
    word_tensors = torch.LongTensor([word_to_index[word] for word in words])
    return word_tensors

def batchify(data, batch_size, use_GPU):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(batch_size, -1).t().contiguous()
    if use_GPU:
        data = data.cuda()
    return data

def get_batch(source, context_size, i, use_GPU, evaluation=False):
    seq_len = min(context_size, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    if use_GPU:
      data, target = data.cuda(), target.cuda()
    return data, target
