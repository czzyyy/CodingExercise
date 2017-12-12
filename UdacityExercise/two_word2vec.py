# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/5_word2vec.ipynb
# http://www.thushv.com/natural_language_processing/word2vec-part-1-nlp-with-deep-learning-with-tensorflow-skip-gram/
# http://www.hankcs.com/ml/cbow-word2vec.html
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
import urllib
import pickle
from matplotlib import pylab
from six.moves import range
from sklearn.manifold import TSNE


def maybe_download(filename, filepath, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filepath + filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filepath + filename)
    statinfo = os.stat(filepath + filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?'
        )
    return filename


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


# Build the dictionary and replace rare words with UNK token.
def build_dataset(words):
    count = [['UNK', -1]]
    # most_common(n)返回一个TopN列表。如果n没有被指定，则返回所有元素。
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()  # 按照words中的顺序，保存其在dictionary中的id
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # UNK
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # skip_window的位置就是窗口的中心位置，就是单词对中的第一个单词的位置
        target_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in target_to_avoid:
                target = random.randint(0, span - 1)
            target_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


# Train a skip-gram model.
def main():
    final_embeddings = None
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a label.
    # We pick a random validation set to sample nearest neighbors. here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.array(random.sample(range(valid_window), valid_size))
    num_samples = 64  # Number of negative examples to sample.

    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        # Input data
        train_data = tf.placeholder(dtype=tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(value=valid_examples, dtype=tf.int32)

        # Variables.
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        softmax_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size))
        )
        # (from embedding look up)embedding_size * (embedding_size * vocabulary_size) +vocabulary_size
        softmax_biases = tf.Variable(tf.zeros(vocabulary_size))

        # Model
        # Look up embeddings for inputs.
        embed = tf.nn.embedding_lookup(embeddings, train_data)
        # Compute the softmax loss, using a sample of the negative labels each time.
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases,
                                       labels=train_labels, inputs=embed, num_sampled=num_samples,
                                       num_classes=vocabulary_size)
        )

        # Optimizer.
        # Note: The optimizer will optimize the softmax_weights AND the embeddings.
        # This is because the embeddings are defined as a variable quantity and the
        # optimizer's `minimize` method will by default modify all variable quantities
        # that contribute to the tensor it is passed.
        # See docs on `tf.train.Optimizer.minimize()` for more details.
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

        # Compute the similarity between minibatch examples and all embeddings.
        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

    # train
    num_steps = 100001
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        print('Initialized')
        average_loss = 0
        for step in range(num_steps):
            batch_data, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            _, l = sess.run([optimizer, loss], feed_dict={train_data: batch_data, train_labels: batch_labels})
            average_loss += l
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0
            # note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    # ??? sim[i, :] ???
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
        final_embeddings = normalized_embeddings.eval()
    return final_embeddings


def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15, 15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    pylab.show()


def save_embeddings_pickle(savepath, embes, dict):
    pickle_file = os.path.join(savepath)
    if embes.shape[0] != len(dict):
        raise Exception('embes length != dict length!')
    try:
        f = open(pickle_file, 'wb')
        save = {
            'words_dict': dict,
            'words_embedding': embes,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        statinfo = os.stat(pickle_file)
        print('Compressed pickle size:', statinfo.st_size)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise


if __name__ == '__main__':
    url = 'http://mattmahoney.net/dc/'
    filepath = 'F:/python_code/text/'
    vocabulary_size = 50000
    data_index = 0
    filename = maybe_download('text8.zip', filepath, 31344016)
    words = read_data(filepath + filename)
    print('Data size %d' % len(words))
    data, count, dictionary, reverse_dictionary = build_dataset(words)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    del words  # Hint to reduce memory.
    print('data:', [reverse_dictionary[di] for di in data[:8]])
    for num_skips, skip_window in [(2, 1), (4, 2)]:
        data_index = 0
        batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
        print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
        print('    batch:', [reverse_dictionary[bi] for bi in batch])
        print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

    final_word_embeddings = main()

    num_points = 400
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    two_d_embeddings = tsne.fit_transform(final_word_embeddings[1:num_points + 1, :])

    words = [reverse_dictionary[i] for i in range(1, num_points+1)]
    plot(two_d_embeddings, words)

    # sava data
    save_embeddings_pickle(filepath + 'text8_skip_gram.pickle', final_word_embeddings, dictionary)
