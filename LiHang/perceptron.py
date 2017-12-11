import numpy as np


class Simple_Perceptron(object):
    def __init__(self, learning_rate, train_x, train_y):
        self.learning_rate = learning_rate
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        self.w = np.random.rand(np.shape(train_x)[1])
        self.b = np.random.rand(1)
    def sign(self, x):
        if x > 0:
            return 1
        else:
            return -1

    def update(self, xi, yi):
        self.w = self.w + self.learning_rate * yi * xi
        self.b = self.b + self.learning_rate * yi

    def train(self):
        train_num = np.shape(self.train_x)[0]
        step = 0
        while step < train_num:
            step = 0
            for i in range(train_num):
                xi = self.train_x[i]
                yi = self.train_y[i]
                if yi * (np.matmul(self.w, np.transpose(xi)) + self.b) <= 0:
                    self.update(xi, yi)
                    break
                step = step + 1
        print('train over')

    def test(self, test_x, test_y):
        pred_y = list(map(self.sign, np.matmul(self.w, np.transpose(test_x)) + self.b))
        print('pred_y:', pred_y)
        print('test_y:', test_y)

