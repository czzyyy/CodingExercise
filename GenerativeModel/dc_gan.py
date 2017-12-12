# http://blog.csdn.net/amds123/article/details/54604038
# https://github.com/lpty/tensorflow_tutorial/tree/master/avatarDcgan
# https://github.com/czzyyy/DCGAN-tensorflow

import tensorflow as tf
import numpy as np
import math
import os
import scipy.misc

class DC_GAN(object):
    def __init__(self, learning_rate, noise_size, input_size,
                 training_epochs, batch_size, display_step, size=64):
        self.train_data = None
        self.learning_rate = learning_rate
        self.noise_size = noise_size
        self.input_size = input_size  # [h , w , 通道]
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.display_step = display_step
        self.chunk_size = None  # 一共的batch数目
        self.size = size  # 卷积和解卷积输出通道数量
        self.batch_index = 0

    # 用于计算卷积之后的图像尺寸
    @staticmethod
    def conv_out_size_same(size, stride):
        return int(math.ceil(float(size) / float(stride)))

    # 注意用gan的时候变量声明要用get_variable
    @staticmethod
    def full_connect(x, output_size, stddev=0.02, bias=0.0, name='full_connect'):
        with tf.variable_scope(name):
            shape = x.shape.as_list()  # 不这么写就报错
            w = tf.get_variable('w', [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [output_size],  tf.float32, tf.constant_initializer(bias))
            return tf.matmul(x, w) + b

    @staticmethod
    def batch_normalizer(x, epsilon=1e-5, momentum=0.9, train=True, name='batch_norm'):
        with tf.variable_scope(name):
            return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon,
                                                scale=True, is_training=train)

    @staticmethod
    def conv2d(x, output_size, stddev=0.02, name='conv2d'):
        with tf.variable_scope(name):
            # filter : [height, width, in_channels, output_channels]
            # 注意与解卷积的不同
            shape = x.shape.as_list()
            filter_shape = [5, 5, shape[-1], output_size]
            strides_shape = [1, 2, 2, 1]
            w = tf.get_variable('w', filter_shape, tf.float32, tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [output_size], tf.float32, tf.constant_initializer(0.0))
            return tf.nn.bias_add(tf.nn.conv2d(x, w, strides=strides_shape, padding='SAME'), b)

    @staticmethod
    def deconv2d(x, output_size, stddev=0.02, name='deconv2d'):
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            # 注意与卷积的不同
            shape = x.shape.as_list()
            filter_shape = [5, 5, output_size[-1], shape[-1]]
            strides_shape = [1, 2, 2, 1]
            w = tf.get_variable('w', filter_shape, tf.float32, tf.random_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [output_size[-1]], tf.float32, tf.constant_initializer(0.0))
            return tf.nn.bias_add(tf.nn.conv2d_transpose(x, filter=w, output_shape=output_size,
                                                         strides=strides_shape, padding='SAME'), b)

    @staticmethod
    def resize_img(img, resize_h, resize_w):
        return scipy.misc.imresize(img, [resize_h, resize_w])

    @staticmethod
    def lrelu(x, leak=0.2):
        return tf.maximum(x, leak * x)

    def load_img(self, folder):
        image_files = os.listdir(folder)
        dataset = np.ndarray(
            shape=[len(image_files), ] + self.input_size,
            dtype=np.float32
        )
        num_images = 0
        for image in image_files:
            image_file = os.path.join(folder, image)
            im = scipy.misc.imread(image_file).astype(np.float32)
            if im.shape != self.input_size:
                #print('resize img')
                im = self.resize_img(im, self.input_size[0], self.input_size[1])
            image_data = (np.array(im).astype(np.float32) / (255.0 / 2.0) - 1.0)
            dataset[num_images, :, :, :] = np.reshape(image_data, newshape=self.input_size)
            num_images = num_images + 1
            print(num_images)
            if num_images == 35000:
                break
        dataset = dataset[0:num_images, :, :, :]
        self.chunk_size = int(math.ceil(float(num_images) / float(self.batch_size)))
        print('Chunk_size:', self.chunk_size)
        print('Full dataset tensor:', dataset.shape)
        self.train_data = dataset

    def get_batches(self):
        batch = self.train_data[self.batch_index:self.batch_index + self.batch_size, :, :, :]
        self.batch_index = (self.batch_index + self.batch_size) % ((self.chunk_size - 1) * self.batch_size)
        return batch

    # generate (model 1)
    def build_generator(self, noise, train=True, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            # 分别对应每个layer的height, width
            s_h, s_w, _ = self.input_size
            s_h2, s_w2 = self.conv_out_size_same(s_h, 2), self.conv_out_size_same(s_w, 2)
            s_h4, s_w4 = self.conv_out_size_same(s_h2, 2), self.conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = self.conv_out_size_same(s_h4, 2), self.conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = self.conv_out_size_same(s_h8, 2), self.conv_out_size_same(s_w8, 2)

            # 对输入噪音图片进行线性变换 AttributeError: 'tuple' object has no attribute 'as_list
            z = self.full_connect(noise, self.size * 8 * s_h16 * s_w16, name='g_full')
            # reshape成图像的格式
            h0 = tf.reshape(z, [-1, s_h16, s_w16, self.size * 8])
            # 对数据进行归一化处理 加快收敛速度
            h0 = self.batch_normalizer(h0, train=train, name='g_bn0')
            h0 = tf.nn.relu(h0, name='g_l1')

            h1 = self.deconv2d(h0, output_size=[self.batch_size, s_h8, s_w8, self.size * 4], name='g_h1')
            h1 = self.batch_normalizer(h1, train=train, name='g_bn1')
            h1 = tf.nn.relu(h1, name='g_l1')

            h2 = self.deconv2d(h1, output_size=[self.batch_size, s_h4, s_w4, self.size * 2], name='g_h2')
            h2 = self.batch_normalizer(h2, train=train, name='g_bn2')
            h2 = tf.nn.relu(h2, name='g_l2')

            h3 = self.deconv2d(h2, output_size=[self.batch_size, s_h2, s_w2, self.size * 1], name='g_h3')
            h3 = self.batch_normalizer(h3, train=train, name='g_bn3')
            h3 = tf.nn.relu(h3, name='g_l3')

            h4 = self.deconv2d(h3, output_size=[self.batch_size, ] + self.input_size, name='g_h4')
            x_generate = tf.nn.tanh(h4, name='g_l4')

            return x_generate

    # discriminator (model 2)
    def build_discriminator(self, imgs, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            # 分别对应每个layer的height, width
            s_h, s_w, _ = self.input_size
            s_h2, s_w2 = self.conv_out_size_same(s_h, 2), self.conv_out_size_same(s_w, 2)
            s_h4, s_w4 = self.conv_out_size_same(s_h2, 2), self.conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = self.conv_out_size_same(s_h4, 2), self.conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = self.conv_out_size_same(s_h8, 2), self.conv_out_size_same(s_w8, 2)
            # 卷积操作
            h0 = self.conv2d(imgs, self.size, name='d_h0')
            h0 = self.lrelu(h0)

            h1 = self.conv2d(h0, self.size * 2, name='d_h1')
            h1 = self.batch_normalizer(h1, name='d_bn1')
            h1 = self.lrelu(h1)

            h2 = self.conv2d(h1, self.size * 4, name='d_h2')
            h2 = self.batch_normalizer(h2, name='d_bn2')
            h2 = self.lrelu(h2)

            h3 = self.conv2d(h2, self.size * 8, name='d_h3')
            h3 = self.batch_normalizer(h3, name='d_bn3')
            h3 = self.lrelu(h3)

            h4 = tf.reshape(h3, [self.batch_size, s_h16 * s_w16 * self.size * 8])

            h4 = self.full_connect(h4, 1, name='d_full')
            y_data = tf.nn.sigmoid(h4, name='d_l4')

            return y_data

    def generate_samples(self, num):
        noise_imgs = tf.placeholder(tf.float32, [None, self.noise_size], name='noise_imgs')
        sample_imgs = self.build_generator(noise_imgs, train=False, reuse=True)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, '/root/dcgan.ckpt')
            sample_noise = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.noise_size))
            samples = sess.run(sample_imgs, feed_dict={noise_imgs: sample_noise})[:num]
        for i in range(len(samples)):
            scipy.misc.imsave('F:/tf_board/dc_gan/' + '-' + str(i) + 'generate.png', samples[i])
        print('generate done!')

    def train(self):
        with tf.name_scope('inputs'):
            real_imgs = tf.placeholder(tf.float32, [None, ] + self.input_size, name='real_images')
            noise_imgs = tf.placeholder(tf.float32, [None, self.noise_size], name='noise_images')

        # 生成器图片
        fake_imgs = self.build_generator(noise_imgs)
        # 判别器
        real_logits = self.build_discriminator(real_imgs)
        fake_logits = self.build_discriminator(fake_imgs, reuse=True)

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
            # 所有定义变量
            train_vars = tf.trainable_variables()
            # 生成器变量
            gen_vars = [var for var in train_vars if var.name.startswith('generator')]
            # 判别器变量
            dis_vars = [var for var in train_vars if var.name.startswith('discriminator')]
            # 两个模型的优化函数
            d_trainer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(d_loss, var_list=dis_vars)
            g_trainer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(g_loss, var_list=gen_vars)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            # merge summary
            merged = tf.summary.merge_all()
            # choose dir
            writer = tf.summary.FileWriter('F:/tf_board/dc_gan', sess.graph)
            sess.run(tf.global_variables_initializer())
            for e in range(self.training_epochs):
                check_imgs = None
                for batch_i in range(self.chunk_size):
                    batch_data = self.get_batches()
                    # noise
                    noise = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.noise_size)).astype(np.float32)

                    # Run optimizers
                    sess.run(d_trainer, feed_dict={real_imgs: batch_data, noise_imgs: noise})
                    sess.run(g_trainer, feed_dict={noise_imgs: noise})
                    check_imgs, _ = sess.run([fake_imgs, g_trainer], feed_dict={noise_imgs: noise})

                    if (self.chunk_size * e + batch_i) % self.display_step == 0:
                        train_loss_d = sess.run(d_loss, feed_dict={real_imgs: batch_data, noise_imgs: noise})
                        fake_loss_d = sess.run(d_fake_loss, feed_dict={noise_imgs: noise})
                        real_loss_d = sess.run(d_real_loss, feed_dict={real_imgs: batch_data})
                        # generator loss
                        train_loss_g = sess.run(g_loss, feed_dict={noise_imgs: noise})

                        merge_result = sess.run(merged, feed_dict={real_imgs: batch_data, noise_imgs: noise})
                        # merge_result = sess.run(merged, feed_dict={X: batch_xs})
                        writer.add_summary(merge_result, self.chunk_size * e + batch_i)

                        print("step {}/of epoch {}/{}...".format(self.chunk_size * e + batch_i, e,self.training_epochs),
                              "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(
                                  train_loss_d,real_loss_d,fake_loss_d), "Generator Loss: {:.4f}".format(train_loss_g))

                        # show pic
                        show_imgs = check_imgs[:2]
                        scipy.misc.imsave('F:/tf_board/dc_gan/' + str(self.chunk_size * e + batch_i) +
                                          '-' + str(0) + '.png', show_imgs[0])
                        scipy.misc.imsave('F:/tf_board/dc_gan/' + str(self.chunk_size * e + batch_i) +
                                          '-' + str(1) + '.png', show_imgs[1])

            print('train done')
            # save sess
            saver.save(sess, '/root/dcgan.ckpt')

if __name__ == '__main__':
    data_folder = 'F:/python_code/images/faces'
    dcgan = DC_GAN(learning_rate=0.0002, noise_size=100, input_size=[48, 48, 3], training_epochs=20,
                   batch_size=64, display_step=128, size=64)
    dcgan.load_img(data_folder)
    dcgan.train()
    dcgan.generate_samples(5)