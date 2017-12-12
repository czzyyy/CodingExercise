# 参考：http://blog.csdn.net/wds2006sdo/article/details/53084871
import numpy as np
import csv
import random


class Logistic_Regression(object):
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
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

    def sigmoid(self, x):
        tmp = np.array(np.exp(-x))
        return np.array(1.0 / (tmp + 1.0))

    def cal_grad(self, x, y):
        return np.array(np.array(y - self.sigmoid(self.weights.dot(np.transpose(x)))).dot(x)) \
               * (-1.0 / x.shape[0])

    def train(self):
        self.weights = np.zeros(len(self.train_x[0]))
        #这里直接用全部数据进行梯度计算了，可以先计算概率，将误分类的样本进行梯度计算和迭代(sgd)
        for i in range(self.max_iter):
            grad = self.cal_grad(self.train_x, self.train_y)
            print('grad_max:', np.max(grad))
            print('training_iter:', i)
            self.weights = self.weights - self.learning_rate * grad
            print('weights_max:', np.max(self.weights))
        print('train over')

    def test(self):
        result = 0
        for i in range(len(self.test_y)):
            x = self.test_x[i]
            pred1 = self.sigmoid(self.weights.dot(np.transpose(x)))
            pred0 = 1.0 - pred1
            if pred0 > pred1:
                pred = 0
            else:
                pred = 1
            if int(pred) == int(self.test_y[i]):
                result = result + 1
        print('acc:', float(result) / float(len(self.test_y)))

    def predict(self, x):
        pred1 = self.sigmoid(self.weights.dot(np.transpose(x)))
        pred0 = 1.0 - pred1
        if pred0 > pred1:
            print('pred:', 0)
        else:
            print('pred:', 1)
   
#简单使用
# import LiHang.logistics_regression as lr
# a_logistic_reg = lr.Logistic_Regression(0.01, 1000)
# datasets = a_logistic_reg.load_csv('data/train_binary.csv')
# a_logistic_reg.split_dataset(datasets, 0.7)
# for i in range(len(a_logistic_reg.train_x)):
#     for j in range(len(a_logistic_reg.train_x[i])):
#         if a_logistic_reg.train_x[i][j] > 30.0:
#             a_logistic_reg.train_x[i][j] = 1
#         else:
#             a_logistic_reg.train_x[i][j] = 0
#
# for i in range(len(a_logistic_reg.test_x)):
#     for j in range(len(a_logistic_reg.test_x[i])):
#         if a_logistic_reg.test_x[i][j] > 30.0:
#             a_logistic_reg.test_x[i][j] = 1
#         else:
#             a_logistic_reg.test_x[i][j] = 0
# a_logistic_reg.train()
# a_logistic_reg.test()
