import numpy as np
import csv
import random


class Softmax_Regression(object):
    def __init__(self, learning_rate, max_iter, class_num):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.class_num = class_num
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    def load_csv(self, filename):
        lines = csv.reader(open(filename, 'r'))
        datasets = list(lines)
        if isinstance(datasets[0][0], str):
            datasets = datasets[1:len(datasets)]
        for i in range(len(datasets)):
            datasets[i] = [float(x) for x in datasets[i]]
        print('load over')
        return datasets

    def split_dataset(self, dataset, train_ratio):
        train_size = int(len(dataset) * train_ratio)
        train_set = []
        copy = list(dataset)
        while len(train_set) < train_size:
            index = random.randrange(len(copy))
            train_set.append(copy.pop(index))
        self.train_x = np.array([i[1:len(i)] for i in train_set])
        self.train_y = np.array([i[0] for i in train_set])
        self.test_x = np.array([i[1:len(i)] for i in copy])
        self.test_y = np.array([i[0] for i in copy])

    def cal_e(self, x, l):
        theta_l = self.weights[l]
        product = np.dot(theta_l, x)
        return np.exp(product)

    def cal_probability(self, x, j):
        molecule = self.cal_e(x, j)
        denominator = sum([self.cal_e(x, i) for i in range(self.class_num)])
        return molecule/denominator

    def cal_partial_derivative(self, x, y, j):
        first = int(y == j)
        second = self.cal_probability(x, j)
        return -x*(first-second) + 0.01*self.weights[j]

    def train(self):
        self.weights = np.zeros([self.class_num, len(self.train_x[0]) + 1])
        for i in range(self.max_iter):
            index = random.randrange(len(self.train_y))
            x = self.train_x[index]
            x = list(x)
            x.append(1.0)
            x = np.array(x)
            y = self.train_y[index]
            derivatives = [self.cal_partial_derivative(x, y, j) for j in range(self.class_num)]
            for k in range(self.class_num):
                self.weights[k] -= self.learning_rate * derivatives[k]
            print('training_iter:', i)
            print('min_weights:', np.min(self.weights))
        print('train over')

    def test(self):
        result = 0
        for i in range(len(self.test_y)):
            x = self.train_x[i]
            x = list(x)
            x.append(1.0)
            x = np.array(x)
            max_pred = 0
            pred = 0
            for j in range(self.class_num):
                preds = self.cal_probability(x, j)
                if preds > max_pred:
                    max_pred = preds
                    pred = j
            if int(pred) == int(self.train_y[i]):
                print('label:', int(pred))
                result = result + 1
        print('acc:', float(result) / float(len(self.test_y)))

    def predict(self, x):
        x = list(x)
        x.append(1.0)
        x = np.array(x)
        max_pred = 0
        pred = 0
        for i in range(self.class_num):
            preds = self.cal_probability(x, i)
            if preds > max_pred:
                max_pred = preds
                pred = i
        print('pred:', pred)
  
#简单使用
# import LiHang.softmax_regression as sr
# a_softmax_reg = sr.Softmax_Regression(0.000001, 200000, 10)
# datasets = a_softmax_reg.load_csv('data/train.csv')
# a_softmax_reg.split_dataset(datasets, 0.7)
# a_softmax_reg.train()
# a_softmax_reg.test()
#
# import load_img_for_mnist as load
# im_test = load.imageprepare('some_pic_like_three.png')
# print('shape:', len(im_test))
# print('img_data:', im_test)
#
# a_softmax_reg.predict(im_test)
