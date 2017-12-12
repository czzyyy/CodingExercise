# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb
# http://www.hankcs.com/ml/notmnist.html
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import random
from sklearn import  linear_model
from PIL import Image
import LiHang.softmax_regression as sr
from scipy import ndimage
from six.moves import cPickle as pickle

num_classes = 10
np.random.seed(133)
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
train_size = 200000
valid_size = 10000
test_size = 10000


# extract the tar
def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar .gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already exists-Skipping extraction of %s', (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
    ]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders))
        )
    print(data_folders)
    return data_folders


# A few images might not be readable, we'll just skip them.
def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(
        shape=(len(image_files), image_size, image_size),
        dtype=np.float32
    )
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]  #????
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


def randomize(dataset, lables):
    permutation = np.random.permutation(lables.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = lables[permutation]
    return shuffled_dataset, shuffled_labels


def sava_all_pickle(train_d, train_l, valid_d, valid_l, test_d, test_l):
    pickle_file = os.path.join(data_root, 'notMNIST.pickle')
    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_d,
            'train_labels': train_l,
            'valid_dataset': valid_d,
            'valid_labels': valid_l,
            'test_dataset': test_d,
            'test_labels': test_l,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        statinfo = os.stat(pickle_file)
        print('Compressed pickle size:', statinfo.st_size)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise


def pro1_sample(data_folders, sample_size, title=None):
    fig = plt.figure()
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    for folder in data_folders:
        image_files = os.listdir(folder)
        image_sample = random.sample(image_files, sample_size)
        for image in image_sample:
            image_file = os.path.join(folder, image)
            ax = fig.add_subplot(len(data_folders), sample_size, sample_size *
                                 data_folders.index(folder) + image_sample.index(image) + 1)
            image_data = Image.open(image_file)
            ax.imshow(image_data)
            ax.set_axis_off()
    plt.show()


def pro2_load_display_pickle(datasets, sample_size, title=None):
    fig = plt.figure()
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    num_of_images = []
    for pickle_file in datasets:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            print('Total images in', pickle_file, ':', len(data))

            for index, image in enumerate(data):
                if index == sample_size: break
                ax = fig.add_subplot(len(datasets), sample_size,
                                     sample_size * datasets.index(pickle_file) +
                                     index + 1)
                ax.imshow(image)
                ax.set_axis_off()
                ax.imshow(image)
            num_of_images.append(len(data))

    plt.show()
    pro3_balance_check(num_of_images)
    return num_of_images


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def pro3_balance_check(sizes):
    plt.figure()
    mean_val = mean(sizes)
    print('mean of # images :', mean_val)
    for i in sizes:
        if abs(i - mean_val) > 0.1 * mean_val:
            print('Too much or less images')
        else:
            print('Well balanced', i)
    # draw bar
    index = range(10)
    plt.bar(index, sizes)
    plt.show()


def pro4_sample_shuffle(dataset, labels, title=None):
    plt.figure()
    if title:
        plt.suptitle(title, fontsize=16, fontweight='bold')
    items = random.sample(range(len(labels)), 12)
    for i, item in enumerate(items):
        plt.subplot(3, 4, i+1)
        plt.axis('off')
        plt.title(chr(ord('A') + labels[item]))
        plt.imshow(dataset[item])
    plt.show()


if __name__ == '__main__':
    data_root = 'F:/python_code/not_MNIST/'  # Change me to store data elsewhere
    train_filename = 'F:/python_code/not_MNIST/notMNIST_large.tar.gz'
    test_filename = 'F:/python_code/not_MNIST/notMNIST_small.tar.gz'
    train_folders = maybe_extract(train_filename)
    test_folders = maybe_extract(test_filename)
    # pro1_sample(train_folders, 10, 'Train Folders')
    # pro1_sample(test_folders, 10, 'Test Folders')
    train_datasets = maybe_pickle(train_folders, 45000)
    test_datasets = maybe_pickle(test_folders, 1800)
    # pro2_load_display_pickle(train_datasets, 10, 'Train Datasets')
    # pro2_load_display_pickle(test_datasets, 10, 'Test Datasets')
    # valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    #     train_datasets, train_size, valid_size)
    # _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)
    #
    # print('Training:', train_dataset.shape, train_labels.shape)
    # print('Validation:', valid_dataset.shape, valid_labels.shape)
    # print('Testing:', test_dataset.shape, test_labels.shape)
    #
    # train_dataset, train_labels = randomize(train_dataset, train_labels)
    # test_dataset, test_labels = randomize(test_dataset, test_labels)
    # valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
    #
    # pro4_sample_shuffle(train_dataset, train_labels, 'train dataset suffled')
    # pro4_sample_shuffle(valid_dataset, valid_labels, 'valid dataset suffled')
    # pro4_sample_shuffle(test_dataset, test_labels, 'test dataset suffled')
    #
    # sava_all_pickle(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
    pickle_file_path = os.path.join(data_root, 'notMNIST.pickle')
    a_softmax_reg = sr.Softmax_Regression(0.001, 150000, 10)
    with open(pickle_file_path, 'rb') as f:
        pickle_file = pickle.load(f)
        X_train = pickle_file['train_dataset'][:10000].reshape(10000, image_size * image_size)
        y_train = pickle_file['train_labels'][:10000]
        X_test = pickle_file['test_dataset'].reshape(pickle_file['test_dataset'].shape[0], image_size * image_size)
        y_test = pickle_file['test_labels']
        # reg = linear_model.LogisticRegression()
        # reg.fit(X_train, y_train)
        # pred_labels = reg.predict(X_test)
        # print('Accuracy:', reg.score(X_test, y_test))
        a_softmax_reg.train_x = X_train
        a_softmax_reg.train_y = y_train
        a_softmax_reg.test_x = X_test
        a_softmax_reg.test_y = y_test
        a_softmax_reg.train()
        a_softmax_reg.test()


