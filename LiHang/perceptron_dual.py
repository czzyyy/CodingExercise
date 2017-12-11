import numpy as np


class Simple_Perceptron_Dual(object):
    def __init__(self, learning_rate, train_x, train_y):
        self.learning_rate = learning_rate
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        self.alpha = np.zeros([1, np.shape(train_x)[0]])
        self.w = np.random.rand(1, np.shape(train_x)[1])
        self.b = np.zeros([1])

    def sign(self, x):
        if x > 0:
            return 1
        else:
            return -1

    def update(self, yi, i):
        self.alpha[0][i] = self.alpha[0][i] + self.learning_rate
        self.b = self.b + self.learning_rate * yi

    def train(self):
        G = np.matmul(self.train_x, np.transpose(self.train_x))
        train_num = np.shape(self.train_x)[0]
        step = 0
        while step < train_num:
            step = 0
            for i in range(train_num):
                xi = self.train_x[i]
                yi = self.train_y[i]
                if yi * (np.sum(self.alpha * self.train_y * G[:, i]) + self.b) <= 0:
                    self.update(yi, i)
                    break
                step = step + 1
        self.w = np.matmul(self.train_y * self.alpha, self.train_x)
        print('train over')

    def test(self, test_x, test_y):
        pred_y = list(map(self.sign, np.matmul(self.w[0], np.transpose(test_x)) + self.b))
        print('pred_y:', pred_y)
        print('test_y:', test_y)

