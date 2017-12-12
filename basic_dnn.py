from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# parameters
learning_rate = 0.001
training_iters = 10000
batch_size = 100
display_step = 100

# Network para
n_input = 784
n_classes = 10
dropout = 0.80

# weights biases
weights = {
    'w1': tf.Variable(tf.random_normal([784, 600])),
    'w2': tf.Variable(tf.random_normal([600, 450])),
    'w3': tf.Variable(tf.random_normal([450, 300])),
    'out': tf.Variable(tf.random_normal([300, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([600])),
    'b2': tf.Variable(tf.random_normal([450])),
    'b3': tf.Variable(tf.random_normal([300])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dnn(_X, _Weights, _Biases, _Dropout):
    # _X = tf.nn.dropout(_X, _Dropout)
    d1 = tf.nn.relu(tf.matmul(_X, _Weights['w1']) + _Biases['b1'])
    # x2 = tf.nn.dropout(d1, _Dropout)
    d2 = tf.nn.relu(tf.matmul(d1, _Weights['w2']) + _Biases['b2'])
    d3 = tf.nn.relu(tf.matmul(d2, _Weights['w3']) + _Biases['b3'])
    x4 = tf.nn.dropout(d3, _Dropout)
    out = tf.matmul(x4, _Weights['out']) + _Biases['out']
    return out


def main(argv):
    # Import data
    mnist = input_data.read_data_sets(argv, one_hot=True)

    # x, y, prob
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    y_pred = dnn(x, weights, biases, keep_prob)
    # loss optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # eval
    correct = tf.equal(tf.argmax(y, 1), tf.argmax((y_pred, 1)))
    accuracy = tf.reduce_mean(tf.cast(correct, "float"))

    # saver
    saver = tf.train.Saver()
    max_acc = 0.98
    max_sess = None
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(training_iters):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            if i % display_step == 0:
                tra_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                tra_loss = sess.run(loss, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                print('iter: %d, minibatch_acc: %g, minibatch_loss: %g' % (i, tra_acc, tra_loss))
                # save sess
                if tra_acc > max_acc:
                    max_acc = tra_acc
                    max_sess = sess
        print('training over!')
        # save sess
        saver.save(max_sess, '/root/mnistnet.ckpt')
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
    mnist_save_path = "/tmp/data/"
    main(mnist_save_path)
