# aae的测试
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
img_height = 28
img_width = 28
img_size = img_height * img_width

# 总迭代次数
max_epoch = 5
learning_rate = 0.001
h_size = 16
batch_size = 64
display_step = 200
generate_num = 10
test_label = np.argmax(mnist.test.labels, 1)
test_data = mnist.test.images


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.02)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


# without popling downsample by strides
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding='SAME')


def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak * x, name=name)


def build_encoder(x):
    with tf.name_scope('encoder'):
        x_image = tf.reshape(x, [-1, img_height, img_width, 1])

        W_conv1 = weight_variable([5, 5, 1, 16], 'W_conv1')
        b_conv1 = bias_variable([16], 'b_conv1')
        conv1 = lrelu(tf.nn.bias_add(conv2d(x_image, W_conv1), b_conv1), name='conv1')

        W_conv2 = weight_variable([5, 5, 16, 32], 'W_conv2')
        b_conv2 = bias_variable([32], 'b_conv2')
        conv2 = lrelu(tf.nn.bias_add(conv2d(conv1, W_conv2), b_conv2), name='conv2')

        conv2_flat = tf.reshape(conv2, [-1, int(img_height/2/2) * int(img_width/2/2) * 32])

        # full connect layer
        W_full1 = weight_variable([int(img_height/2/2) * int(img_width/2/2) * 32, h_size], 'W_full1')
        b_full1 = bias_variable([h_size], 'b_full1')
        z_mean = tf.nn.bias_add(tf.matmul(conv2_flat, W_full1), b_full1, name='z_mean')
        W_full2 = weight_variable([int(img_height/2/2) * int(img_width/2/2) * 32, h_size], 'W_full2')
        b_full2 = bias_variable([h_size], 'b_full2')
        z_stddev = tf.nn.bias_add(tf.matmul(conv2_flat, W_full2), b_full2, name='z_stddev')

        return z_mean, z_stddev


def build_decoder(z):
    # deconv
    with tf.name_scope('decoder'):
        W_full = weight_variable([h_size, int(img_height/2/2) * int(img_width/2/2) * 32], 'W_full')
        b_full = bias_variable([int(img_height/2/2) * int(img_width/2/2) * 32], 'b_full')

        z_full = tf.nn.bias_add(tf.matmul(z, W_full), b_full, name='z_full')
        z_matrix = tf.nn.relu(tf.reshape(z_full, [batch_size, int(img_height/2/2), int(img_width/2/2), 32]),
                              name='z_matrix')
        W_h1 = weight_variable([5, 5, 16, 32], 'W_h1')
        W_h2 = weight_variable([5, 5, 1, 16], 'W_h2')
        # attention 5 5 16 32 not 5 5 32 16
        h1 = tf.nn.relu(tf.nn.conv2d_transpose(z_matrix, W_h1, [batch_size, int(img_height/2), int(img_width/2), 16],
                                               strides=[1, 2, 2, 1], padding="SAME"), name='h1')

        # attention sigmoid
        h2 = tf.nn.sigmoid(tf.nn.conv2d_transpose(h1, W_h2, [batch_size, img_height, img_width, 1],
                                                  strides=[1, 2, 2, 1], padding="SAME"), name='h2')

        return h2


def gaussian(size, ndim, mean=0, var=1):
    return np.random.normal(mean, var, (size, ndim)).astype(np.float32)


def generate_samples(num):
    input_imgs = tf.placeholder(tf.float32, [None, img_size], name='input_imgs')
    encode_z_mean, encode_z_stddev = build_encoder(input_imgs)
    samples = tf.random_normal([batch_size, h_size], 0, 1, dtype=tf.float32)
    fake_z = tf.add(tf.multiply(samples, encode_z_stddev), encode_z_mean, name='fake_z')
    generated_images = build_decoder(fake_z)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/root/aae_mnist/aae_mnist.ckpt')
        samples = sess.run(generated_images, feed_dict={input_imgs: test_data[0:batch_size]})[:num]
    for i in range(len(samples)):
        plt.imsave('F:/tf_board/aae_mnist/' + str(i) + 'generate.png', samples[i][:,:,0],cmap='Greys_r')
    print('generate done!')

if __name__ == '__main__':
    generate_samples(10)