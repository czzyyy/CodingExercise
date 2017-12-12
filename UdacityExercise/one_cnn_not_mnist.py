# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/2_fullyconnected.ipynb
import numpy as np
import tensorflow as tf
import pickle
from six.moves import range


def reload_pickle(filename):
    with open(filename, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory!!!
        print('training set:', train_dataset.shape, train_labels.shape)
        print('valid set:', valid_dataset.shape, valid_labels.shape)
        print('test_set:', test_dataset.shape, test_labels.shape)
        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


def reformat(dataset, labels, num_labels, image_size):
    dataset = dataset.reshape(dataset.shape[0], image_size * image_size).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    print('data set', dataset.shape, labels.shape)
    return dataset, labels


# define function
def conv2d(x, W, name=None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


def max_pool_kk(x, k, name=None):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def weight_variable(shape, name=None):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def net(x):
    with tf.name_scope('net'):
        # reshape x
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        # First convolutional layer - maps one grayscale image to 32 feature maps.
        W_conv1 = weight_variable([5, 5, 1, 32], 'W_con1')
        b_conv1 = bias_variable([32], 'b_conv1')
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        h_pool1 = max_pool_kk(h_conv1, 2, 'h_pool1')

        # Second convolutional layer - maps one grayscale image to 32 feature maps.
        W_conv2 = weight_variable([5, 5, 32, 64], 'W_con2')
        b_conv2 = bias_variable([64], 'b_conv2')
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # Second Pooling layer - downsamples by 2X.
        h_pool2 = max_pool_kk(h_conv2, 2, 'h_pool2')

        # drop
        keep_prob = tf.placeholder(tf.float32)
        h_pool2 = tf.nn.dropout(h_pool2, keep_prob, name='drop1')

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        W_fc1 = weight_variable([7 * 7 * 64, 1024], 'W_full1')
        b_fc1 = bias_variable([1024], 'b_full1')

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='drop2')

        # Map the 1024 features to 10 classes, one for each digit
        W_fc2 = weight_variable([1024, 10], 'W_full2')
        b_fc2 = bias_variable([10], 'b_full2')

        y_conv = tf.matmul(h_fc1_drop, W_fc2)+b_fc2
        return y_conv, keep_prob


def main():
    pickle_filename = 'F:/python_code/not_MNIST/notMNIST.pickle'
    image_size = 28
    num_labels = 10
    num_steps = 20000
    batch_size = 128
    # load data
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = reload_pickle(pickle_filename)
    train_dataset, train_labels = reformat(train_dataset, train_labels, num_labels, image_size)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, num_labels, image_size)
    test_dataset, test_labels = reformat(test_dataset, test_labels, num_labels, image_size)
    # Create the model
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, [None, image_size * image_size], name='input_x')
        y_ = tf.placeholder(tf.float32, [None, num_labels], name='input_y')
    # Build the graph for the deep net
    y_pred, keep_pro = net(x)

    # define loss and optimizer
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_pred))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # accuracy
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    # saver
    saver = tf.train.Saver()
    max_acc = 0.9
    max_sess = None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # merge summary
        merged = tf.summary.merge_all()
        # choose dir
        writer = tf.summary.FileWriter('F:/tf_board/not_mnist_conv1', sess.graph)
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_x = train_dataset[offset: (offset + batch_size), :]
            batch_label = train_labels[offset: (offset + batch_size), :]
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_label, keep_pro: 0.75})
            if step % 500 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y_: batch_label, keep_pro: 1.0})
                print('step %d, training accuracy %g' % (step, train_accuracy))
                merge_result = sess.run(merged, feed_dict={x: batch_x, y_: batch_label, keep_pro: 1.0})
                writer.add_summary(merge_result, step)
                # save sess
                if train_accuracy > max_acc:
                    max_acc = train_accuracy
                    max_sess = sess
        print('train done')
        # save sess
        saver.save(max_sess, '/root/not_mnistnet.ckpt')
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: test_dataset, y_: test_labels, keep_pro: 1.0}))


if __name__ == '__main__':
    main()
