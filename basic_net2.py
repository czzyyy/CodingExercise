from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# Import data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# parameters
learning_rate = 0.0001
training_iters = 10000
batch_size = 100
display_step = 100

# Network para
n_input = 784
n_classes = 10
dropout = 0.6

# tf graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout

# create model
# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([5, 5, 128, 256])),
    'wc4': tf.Variable(tf.random_normal([5, 5, 256, 512])),
    'wd1': tf.Variable(tf.random_normal([2 * 2 * 512, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bc4': tf.Variable(tf.random_normal([512])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# define conv with relu
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))


# define max_pool
def max_pool(img, k):
    return tf.nn.max_pool(img, [1, k, k, 1], [1, k, k, 1], 'SAME')


def norm(img, k=4):
    return tf.nn.lrn(img, k, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


# define net
def conv_net(_X, _weights, _biases, _dropout):
    # reshape input pic
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])

    # convolution layer
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    # max pooling (down sample)
    pool1 = max_pool(conv1, k=2)
    # Apply Normalization
    norm1 = norm(pool1, k=4)
    # dropout
    drop1 = tf.nn.dropout(norm1, _dropout)

    # convolution layer
    conv2 = conv2d(drop1, _weights['wc2'], _biases['bc2'])
    # max pooling (down sample)
    pool2 = max_pool(conv2, k=2)
    # Apply Normalization
    norm2 = norm(pool2, k=4)
    # dropout
    drop2 = tf.nn.dropout(norm2, _dropout)

    # convolution layer
    conv3 = conv2d(drop2, _weights['wc3'], _biases['bc3'])
    # max pooling (down sample)
    pool3 = max_pool(conv3, k=2)
    # Apply Normalization
    norm3 = norm(pool3, k=4)
    # dropout
    drop3 = tf.nn.dropout(norm3, _dropout)

    # convolution layer
    conv4 = conv2d(drop3, _weights['wc4'], _biases['bc4'])
    # max pooling (down sample)
    pool4 = max_pool(conv4, k=2)
    # Apply Normalization
    norm4 = norm(pool4, k=4)
    # dropout
    drop4 = tf.nn.dropout(norm4, _dropout)

    # Fully connected layer
    dense1 = tf.reshape(drop4, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
    dense1 = tf.nn.dropout(dense1, _dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    return out


# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# saver
saver = tf.train.Saver()
max_acc = 0.9
max_sess = None

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
            if acc > max_acc:
                max_acc = acc
                max_sess = sess
            print("Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1

    # save sess
    saver.save(max_sess, '/root/mnistnet.ckpt')
    print("Optimization Finished!")
    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.0}))


'''
saver = tf.train.Saver()
with tf.Session() as sess:

    saver.restore(sess, '/root/alexnet.tfmodel')
    sess.run(....)
'''