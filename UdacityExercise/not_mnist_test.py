import tensorflow as tf
import numpy as np
from PIL import Image

# predict number
def predictint(imvalue):
    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 784])
    keep_prob = tf.placeholder(tf.float32)

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

    # Build the graph for the deep net
    y_pred, keep_pro = net(x)
    init = tf.global_variables_initializer()

    # load the model
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, '/root/not_mnistnet.ckpt')
        prediction = tf.argmax(y_pred, 1)
        return prediction.eval(feed_dict={x: [imvalue], keep_prob: 1.0}, session=sess)


# main
def main(argv):
    # load the img
    im = Image.open(argv).resize((28, 28), Image.ANTIALIAS).convert('L')
    imvalue = np.array(im.getdata())
    image_data = (imvalue.astype(float) - 255.0 / 2) / 255.0
    print(len(image_data))
    print(image_data)
    predint = predictint(image_data)
    alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    print(alpha[int(predint[0])])  # first value in list


if __name__ == "__main__":
    main('F:/python_code/a1.png')
