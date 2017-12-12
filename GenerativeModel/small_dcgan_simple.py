import tensorflow as tf
import numpy as np
import math
import os
import scipy.misc

learning_rate = 0.0005
noise_size = 100
input_size = [48, 48, 3]
training_epochs = 40
batch_size = 64
display_step = 128
size = 24  # 卷积和解卷积输出通道数量


def load_img(folder):
    image_files = os.listdir(folder)
    dataset = np.ndarray(
        shape=[len(image_files), ] + input_size,
        dtype=np.float32
    )
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        im = scipy.misc.imread(image_file).astype(np.float)
        if im.shape != input_size:
            im = resize_img(im, input_size[0], input_size[1])
        image_data = (np.array(im).astype(float) / (255.0 / 2.0) - 1)
        dataset[num_images, :, :, :] = np.reshape(image_data, newshape=input_size)
        num_images = num_images + 1
        print(num_images)
        if num_images == 20000:
            break
    dataset = dataset[0:num_images, :, :, :]
    chunk_size = int(math.ceil(float(num_images) / float(batch_size)))
    print('Chunk_size:', chunk_size)
    print('Full dataset tensor:', dataset.shape)
    return dataset, chunk_size


# 用于计算卷积之后的图像尺寸
def conv_out_size_same(s, stride):
    return int(math.ceil(float(s) / float(stride)))


def batch_normalizer(x, epsilon=1e-5, momentum=0.9, train=True, name='batch_norm', reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon,
                                            scale=True, is_training=train)


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.02)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


# without popling downsample by strides
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding='SAME')


def deconv2d(x, output_size, w):
    return tf.nn.conv2d_transpose(x, filter=w, output_shape=output_size,
                                  strides=[1, 2, 2, 1], padding='SAME')


def resize_img(img, resize_h, resize_w):
    return scipy.misc.imresize(img, [resize_h, resize_w])


def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak * x, name=name)


def get_batches(data, batch_index):
    batch = data[batch_index:batch_index + batch_size, :, :, :]
    return batch


# 分别对应每个layer的height, width
s_h, s_w, _ = input_size
s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

# variable g
g_w_full = weight_variable(shape=[noise_size, size * 4 * s_h8 * s_w8], name='g_w_full')
g_b_full = bias_variable(shape=[size * 4 * s_h8 * s_w8], name='g_b_full')
g_w_de1 = weight_variable(shape=[5, 5, size * 2, size * 4], name='g_w_de1')
g_b_de1 = bias_variable(shape=[size * 2], name='g_b_de1')
g_w_de2 = weight_variable(shape=[5, 5, size * 1, size * 2], name='g_w_de2')
g_b_de2 = bias_variable(shape=[size * 1], name='g_b_de2')
g_w_de3 = weight_variable(shape=[5, 5, input_size[2], size * 1], name='g_w_de3')
g_b_de3 = bias_variable(shape=[input_size[2]], name='g_b_de3')
g_params = [g_w_full, g_b_full, g_w_de1, g_b_de1, g_w_de2, g_b_de2, g_w_de3, g_b_de3]

# variable d
d_w_de1 = weight_variable(shape=[5, 5, input_size[2], size * 1], name='d_w_de1')
d_b_de1 = bias_variable(shape=[size * 1], name='d_b_de1')
d_w_de2 = weight_variable(shape=[5, 5, size * 1, size * 2], name='d_w_de2')
d_b_de2 = bias_variable(shape=[size * 2], name='d_b_de2')
d_w_de3 = weight_variable(shape=[5, 5, size * 2, size * 4], name='d_w_de3')
d_b_de3 = bias_variable(shape=[size * 4], name='d_b_de3')
d_w_full = weight_variable(shape=[s_h8 * s_w8 * size * 4, 1], name='d_w_full')
d_b_full = bias_variable(shape=[1], name='d_b_full')
d_params = [d_w_de1, d_b_de1, d_w_de2, d_b_de2, d_w_de3, d_b_de3, d_w_full, d_b_full]


# generate (model 1)
def build_generator(noise, train=True, batch_norm_reuse=None):
    with tf.name_scope('generator'):
        # 对输入噪音图片进行线性变换 AttributeError: 'tuple' object has no attribute 'as_list
        z = tf.matmul(noise, g_w_full) + g_b_full
        # reshape成图像的格式
        h1 = tf.reshape(z, [-1, s_h8, s_w8, size * 4])
        # 对数据进行归一化处理 加快收敛速度
        h1 = batch_normalizer(h1, train=train, name='g_bn1', reuse=batch_norm_reuse)
        h1 = tf.nn.relu(h1, name='g_l1')

        h2 = tf.nn.bias_add(deconv2d(h1, [batch_size, s_h4, s_w4, size * 2], g_w_de1), g_b_de1)
        h2 = batch_normalizer(h2, train=train, name='g_bn2', reuse=batch_norm_reuse)
        h2 = tf.nn.relu(h2, name='g_l2')

        h3 = tf.nn.bias_add(deconv2d(h2, [batch_size, s_h2, s_w2, size * 1], g_w_de2), g_b_de2)
        h3 = batch_normalizer(h3, train=train, name='g_bn3', reuse=batch_norm_reuse)
        h3 = tf.nn.relu(h3, name='g_l3')

        # h3 = deconv2d(h2, output_size=[batch_size, s_h2, s_w2, self.size * 1], name='g_h3')
        # h3 = batch_normalizer(h3, train=train, name='g_bn3')
        # h3 = tf.nn.relu(h3, name='g_l3')

        h4 = tf.nn.bias_add(deconv2d(h3, [batch_size, ] + input_size, g_w_de3), g_b_de3)
        x_generate = tf.nn.tanh(h4, name='g_l4')

        return x_generate


# discriminator (model 2)
def build_discriminator(imgs, batch_norm_reuse=False):
    with tf.name_scope('discriminator'):
        # 卷积操作
        h1 = tf.nn.bias_add(conv2d(imgs, d_w_de1), d_b_de1)
        h1 = lrelu(h1, name='d_l1')

        h2 = tf.nn.bias_add(conv2d(h1, d_w_de2), d_b_de2)
        h2 = batch_normalizer(h2, name='d_bn2', reuse=batch_norm_reuse)
        h2 = lrelu(h2, name='d_l2')

        h3 = tf.nn.bias_add(conv2d(h2, d_w_de3), d_b_de3)
        h3 = batch_normalizer(h3, name='d_bn3', reuse=batch_norm_reuse)
        h3 = lrelu(h3, name='d_l3')

        # h3 = conv2d(h2, size * 8, name='d_h3')
        # h3 = batch_normalizer(h3, name='d_bn3')
        # h3 = tf.nn.relu(h3, name='d_l3')

        h4 = tf.reshape(h3, [batch_size, s_h8 * s_w8 * size * 4])

        h4 = tf.matmul(h4, d_w_full) + d_b_full
        y_data = tf.nn.sigmoid(h4, name='d_l4')

        return y_data


def generate_samples(num):
    noise_imgs = tf.placeholder(tf.float32, [None, noise_size], name='noise_imgs')
    sample_imgs = build_generator(noise_imgs, train=False, batch_norm_reuse=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/root/simple_samll_dcgan/small_simple_dcgan.ckpt')
        sample_noise = np.random.uniform(-1.0, 1.0, size=(batch_size, noise_size))
        samples = sess.run(sample_imgs, feed_dict={noise_imgs: sample_noise})[:num]
    for i in range(len(samples)):
        scipy.misc.imsave('F:/tf_board/small_dc_gan_simple/' + str(i) + 'generate.png', samples[i])
    print('generate done!')


def start_train(dataset, chunk_size):
    with tf.name_scope('inputs'):
        real_imgs = tf.placeholder(tf.float32, [None, ] + input_size, name='real_images')
        noise_imgs = tf.placeholder(tf.float32, [None, noise_size], name='noise_images')

    # 生成器图片
    fake_imgs = build_generator(noise_imgs)
    # 判别器
    real_logits = build_discriminator(real_imgs)
    fake_logits = build_discriminator(fake_imgs, batch_norm_reuse=True)

    # 损失函数的设置
    with tf.name_scope('loss'):
        # 生成器希望判别器判断出来的标签为1
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits, labels=tf.ones_like(fake_logits)))
        # 判别器识别生成器图片loss
        # 判别器希望识别出来的标签为0
        d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits, labels=tf.zeros_like(fake_logits)))
        # 判别器识别真实图片loss
        # 判别器希望识别出来的标签为1
        d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_logits, labels=tf.ones_like(real_logits)))
        # 判别器总loss
        d_loss = tf.add(d_fake_loss, d_real_loss)
        tf.summary.scalar('g_loss', g_loss)
        tf.summary.scalar('d_fake_loss', d_fake_loss)
        tf.summary.scalar('d_real_loss', d_real_loss)
        tf.summary.scalar('d_loss', d_loss)
    with tf.name_scope('optimizer'):
        global_step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(learning_rate, global_step, 256, 0.85, staircase=True)
        rate = tf.maximum(rate, 0.00001)
        #下次改成0.00008?,另外beta1小的时候生成的图像细节较好但是整体会有扭曲，大的时候整体不错，但是模糊
        #考虑动态地改变beta1的值
        # beta1 = 0.8 * np.power(0.9, (beta_step % 800))
        # beta1 = np.maximum(beta1, 0.5)
        # 两个模型的优化函数
        d_trainer = tf.train.AdamOptimizer(rate, beta1=0.5).minimize(d_loss, var_list=d_params, global_step=global_step)
        g_trainer = tf.train.AdamOptimizer(rate, beta1=0.5).minimize(g_loss, var_list=g_params, global_step=global_step)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        # merge summary
        merged = tf.summary.merge_all()
        # choose dir
        writer = tf.summary.FileWriter('F:/tf_board/small_dc_gan_simple', sess.graph)
        batch_index = 0  # init index
        sess.run(tf.global_variables_initializer())
        for e in range(training_epochs):
            for batch_i in range(chunk_size):
                batch_data = get_batches(dataset, batch_index)
                batch_index = (batch_index + batch_size) % ((chunk_size - 1) * batch_size)

                # for beta1 adjust
                #beta_step = beta_step + 1

                # noise
                noise = np.random.uniform(-1.0, 1.0, size=(batch_size, noise_size)).astype(np.float32)

                # Run optimizers
                sess.run(d_trainer, feed_dict={real_imgs: batch_data, noise_imgs: noise})
                #sess.run(g_trainer, feed_dict={noise_imgs: noise})
                check_imgs, _ = sess.run([fake_imgs, g_trainer], feed_dict={noise_imgs: noise})

                if (chunk_size * e + batch_i) % display_step == 0:
                    train_loss_d = sess.run(d_loss, feed_dict={real_imgs: batch_data, noise_imgs: noise})
                    fake_loss_d = sess.run(d_fake_loss, feed_dict={noise_imgs: noise})
                    real_loss_d = sess.run(d_real_loss, feed_dict={real_imgs: batch_data})
                    # generator loss
                    train_loss_g = sess.run(g_loss, feed_dict={noise_imgs: noise})

                    merge_result = sess.run(merged, feed_dict={real_imgs: batch_data, noise_imgs: noise})
                    writer.add_summary(merge_result, chunk_size * e + batch_i)

                    print("step {}/of epoch {}/{}...".format(chunk_size * e + batch_i, e,training_epochs),
                          "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(
                              train_loss_d,real_loss_d,fake_loss_d), "Generator Loss: {:.4f}".format(train_loss_g))

                    # show pic
                    scipy.misc.imsave('F:/tf_board/small_dc_gan_simple/' +
                                      str(chunk_size * e + batch_i) + '-' + str(0) + '.png', check_imgs[0])
                    scipy.misc.imsave('F:/tf_board/small_dc_gan_simple/' +
                                      str(chunk_size * e + batch_i) + '-' + str(1) + '.png', check_imgs[1])

        print('train done')
        # save sess
        saver.save(sess, '/root/simple_samll_dcgan/small_simple_dcgan.ckpt')

if __name__ == '__main__':
    data_folder = 'F:/python_code/images/girlface'
    dataimg, chunk_s = load_img(data_folder)
    #beta_s = 0 #用于调整beta1
    start_train(dataimg, chunk_s)
    generate_samples(5)