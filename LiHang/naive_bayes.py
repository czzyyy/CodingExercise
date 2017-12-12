# 参考：http://blog.csdn.net/wds2006sdo/article/details/51967839
import numpy as np
import random
import csv


class Naive_Bayes(object):
    def __init__(self, class_num, feature_value):
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.feature_value = feature_value  # 图像像素的取值，二值图为2(0和1两个取值);也可以为256(0~255)
        self.class_num = class_num  # 分类类别
        self.prior_probability = None
        self.conditional_probability = None

    def load_articels(self, filename):
        pass

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
        self.train_x = list([i[1:len(i)] for i in train_set])
        self.train_y = list([i[0] for i in train_set])
        self.test_x = list([i[1:len(i)] for i in copy])
        self.test_y = list([i[0] for i in copy])

    def train(self):
        feature_num = len(self.train_x[0])  # 特征个数，也就是训练集样本数目
        train_num = len(self.train_y)
        self.prior_probability = np.ones(self.class_num)
        self.conditional_probability = np.ones([self.class_num, feature_num, self.feature_value])
        for i in range(train_num):
            self.prior_probability[int(self.train_y[i])] = self.prior_probability[int(self.train_y[i])] + 1
            for j in range(feature_num):
                self.conditional_probability[int(self.train_y[i])][j][int(self.train_x[i][j])] += 1
            if i % 100 == 0:
                print('training', i)

        for i in range(self.class_num):
            self.conditional_probability[i] = self.conditional_probability[i] / \
                                              float((self.prior_probability[i] + self.feature_value))

        self.prior_probability = self.prior_probability / float(train_num + self.class_num)
        print('train over')

    def calc_probability(self, train_x):
        probability = np.zeros(self.class_num)
        for i in range(self.class_num):
            probability[i] = self.prior_probability[i]
            for j in range(len(train_x)):
                probability[i] = probability[i] * float(self.conditional_probability[i][j][int(train_x[j])])
        return probability

    def test(self):
        result = 0
        for i in range(len(self.test_y)):
            probability = self.calc_probability(self.test_x[i])
            probability = probability * 1000.0
            max_pro = 0
            predict = 0
            for j in range(len(probability)):
                if probability[j] > max_pro:
                    max_pro = probability[j]
                    predict = j
            if predict == int(self.test_y[i]):
                result = result + 1
            if i % 100 == 0:
                print('testing', i)
        print('accuracy:', float(result) / len(self.test_y))

    def predict(self, x):
        probability = self.calc_probability(x)
        probability = probability * 1000.0
        max_pro = 0
        predict = 0
        for j in range(len(probability)):
            if probability[j] > max_pro:
                max_pro = probability[j]
                predict = j
        print('predict class:', predict)


#简单使用
# import LiHang.naive_bayes as nb
# a_naive_bayes = nb.Naive_Bayes(10, 2)
# datasets = a_naive_bayes.load_csv('data/train.csv')
# a_naive_bayes.split_dataset(datasets, 0.7)
#二值处理
# for i in range(len(a_naive_bayes.train_x)):
#     for j in range(len(a_naive_bayes.train_x[i])):
#         if a_naive_bayes.train_x[i][j] > 30.0:
#             a_naive_bayes.train_x[i][j] = 1
#         else:
#             a_naive_bayes.train_x[i][j] = 0
#
# for i in range(len(a_naive_bayes.test_x)):
#     for j in range(len(a_naive_bayes.test_x[i])):
#         if a_naive_bayes.test_x[i][j] > 30.0:
#             a_naive_bayes.test_x[i][j] = 1
#         else:
#             a_naive_bayes.test_x[i][j] = 0
# a_naive_bayes.train()
# a_naive_bayes.test()
