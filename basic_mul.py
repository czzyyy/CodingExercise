# 参考： https://my.oschina.net/yilian/blog/664087
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# para
n_input = 784
n_classes = 10
n_hidden1 = 256
n_hidden2 = 256

learning_rate = 0.01
batch_size = 100
display_step = 1
training_epochs = 15

# weights bias
weights = {
    'hw1': tf.Variable(tf.random_normal([n_input, n_hidden1]), name='hw1'),
    'hw2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2]), name='hw2'),
    'out': tf.Variable(tf.random_normal([n_hidden2, n_classes]), name='out_w')
}

bias = {
    'hb1': tf.Variable(tf.random_normal([n_hidden1]), name='hb1'),
    'hb2': tf.Variable(tf.random_normal([n_hidden2]), name='hb2'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='out_b')
}


def multilayer(_X, _weights, _biases):
    with tf.name_scope('multilayer'):
        layer1 = tf.nn.relu(tf.matmul(_X, _weights['hw1']) + _biases['hb1'], name='layer1')
        layer2 = tf.nn.relu(tf.matmul(layer1, _weights['hw2']) + _biases['hb2'], name='layer2')
        return tf.add(tf.matmul(layer2, _weights['out']), _biases['out'], name='y_predict')


def main(argv):
    # import data
    mnist = input_data.read_data_sets(argv, one_hot=True)
    # x y
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, [None, n_input], name='x_input')
        y = tf.placeholder(tf.float32, [None, n_classes], name='y_input')
    # predict
    y_pred = multilayer(x, weights, bias)
    # loss & optimizer
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # accuracy
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('accuracy', accuracy)
    # saver
    saver = tf.train.Saver()
    max_acc = 0.9
    max_sess = None
    # run session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # merge summary
        merged = tf.summary.merge_all()
        # choose dir
        writer = tf.summary.FileWriter('F:/tf_board', sess.graph)
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(mnist.train.num_examples/batch_size)
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # fit
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                # loss
                avg_cost = avg_cost + sess.run(loss, feed_dict={x: batch_x, y: batch_y})/total_batch
                # acc
                acc = accuracy.eval({x: batch_x, y: batch_y})
                if (epoch * total_batch + i) % 100 == 0:
                    merge_result = sess.run(merged, feed_dict={x: batch_x, y: batch_y})
                    writer.add_summary(merge_result, (epoch * total_batch + i))
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost), "acc=",
                      "{:.9f}".format(acc))
                if acc > max_acc:
                    max_acc = acc
                    max_sess = sess

        print("Optimization Finished!")
        saver.save(max_sess, '/root/multilayer_net.ckpt')
        test_acc = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
        print("Accuracy:", test_acc)

if __name__ == '__main__':
    save_path = "/tmp/data/"
    main(save_path)
