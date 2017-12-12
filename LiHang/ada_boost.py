# encoding=utf-8
# 参考：http://blog.csdn.net/wds2006sdo/article/details/53195725
import numpy as np
import pandas as pd
import random


class Sign(object):
    def __init__(self, features, labels, w):
        '''
        阈值分类器
        有两种方向，
            1）x<v y=1
            2) x>v y=1
            v 是阈值轴

        因为是针对已经二值化后的MNIST数据集，所以v的取值只有3个 {0,1,2} , ps:可以尝试一下不二值化0~255
        '''
        self.X = features               # 训练数据特征
        self.Y = labels                 # 训练数据的标签
        self.N = len(labels)            # 训练数据大小
        self.w = w                      # 训练数据权值分布
        self.indexes = [0, 1, 2]          # 阈值轴可选范围
        self.is_less = None
        self.index = None

    def _train_less_(self):
        '''
        寻找(x<v y=1)情况下的最优v
        '''
        index = -1  # 记录哪个维度下的
        error = 10000
        for i in self.indexes:
            tmp_error = 0
            for j in range(self.N):
                pred = -1
                if self.X[j] < i:
                    pred = 1
                if pred * self.Y[j] < 0:
                    tmp_error += self.w[j]
            if tmp_error < error:
                error = tmp_error
                index = i
        return index, error

    def _train_more_(self):
        '''
        寻找(x>v y=1)情况下的最优v
        '''
        index = -1  # 记录哪个维度下的
        error = 10000 # 就是误差率em
        for i in self.indexes:
            tmp_error = 0
            for j in range(self.N):
                pred = -1
                if self.X[j] > i:
                    pred = 1
                if pred * self.Y[j] < 0:
                    tmp_error += self.w[j]
            if tmp_error < error:
                error = tmp_error
                index = i
        return index, error

    def train(self):
        less_index, less_error = self._train_less_()
        more_index, more_error = self._train_more_()
        if less_error < more_error:
            self.is_less = True
            self.index = less_index
            return less_error
        else:
            self.is_less = False
            self.index = more_index
            return more_error

    def predict(self, feature):
        if self.is_less is True:
            if feature < self.index:
                return 1
            else:
                return -1
        else:
            if feature > self.index:
                return 1
            else:
                return -1


class Simple_Boost(object):
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.n = None                  # 特征维度
        self.N = None                      # 训练集大小
        self.M = 10                                 # 分类器数目
        self.w = None               # 训练集的权值分布
        self.alpha = []                             # 分类器系数  公式8.2
        self.classifier = []                        # (维度，分类器)，针对当前维度的分类器

    def load_csv(self, filename):
        lines = pd.read_csv(filename, header=0)
        datasets = lines.values
        print('load over')
        return datasets

    def split_dataset(self, dataset, train_ratio, flag='raw'):
        # flag = raw  or binary
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
        if flag == 'binary':
            for i in range(len(self.train_x)):
                for j in range(len(self.train_x[i])):
                    if self.train_x[i][j] > 30:
                        self.train_x[i][j] = 1
                    else:
                        self.train_x[i][j] = 0

            for i in range(len(self.test_x)):
                for j in range(len(self.test_x[i])):
                    if self.test_x[i][j] > 30:
                        self.test_x[i][j] = 1
                    else:
                        self.test_x[i][j] = 0
            print('binary over')
        self.n = len(self.train_x[0])                  # 特征维度
        self.N = len(self.train_y)                      # 训练集大小
        self.M = 10                                 # 分类器数目
        self.w = [1.0/self.N]*self.N                # 训练集的权值分布

    def _Zm_(self, index, classifier):
        Zm = 0
        for i in range(self.N):
            Zm += self._Wm_(index, classifier, i)
        return Zm

    def _Wm_(self, index, classifier, i):
        # 不除以Zm 方便计算
        return self.w[i] * np.exp(-self.alpha[-1] * self.train_y[i] * classifier.predict(self.train_x[i][index]))

    def train(self):
        for times in range(self.M):
            best_classifier = (10000, None, None)  # 误差率,针对的特征，分类器
            for i in range(self.n):
                features = self.train_x[:, i]
                classifier = Sign(features, self.test_y, self.w)
                error = classifier.train()
                if error < best_classifier[0]:
                    best_classifier = (error, i, classifier)
                if (times * self.n + i) % 100 == 0:
                    print("train:", times * self.n + i)
            self.classifier.append(best_classifier[1:])

            em = best_classifier[0]  # 分类误差率
            if em == 0:
                self.alpha.append(100)  # 最大权重
            else:
                self.alpha.append(0.5 * np.log((1-em) / em))

            Zm = self._Zm_(best_classifier[1], best_classifier[2])

            for j in range(self.N):
                self.w[j] = self._Wm_(best_classifier[1], best_classifier[2], j) / Zm

    def _predict_(self, features):
        pred = 0.0
        for i in range(self.M):
            index = self.classifier[i][0]
            classifier = self.classifier[i][1]
            pred += self.alpha[i] * classifier.predict(features[index])

        if pred > 0:
            return 1
        else:
            return 0

    def predict(self):
        result = 0
        for i in range(len(self.test_y)):
            pred = self._predict_(self.test_x[i])
            if int(pred) == int(self.test_y[i]):
                result += 1
        print("acc:", float(result) / float(len(self.test_y)))
        
#简单使用
# import LiHang.ada_boost as ab
# a_ada_boost = ab.Simple_Boost()
# datasets = a_ada_boost.load_csv('data/train_binary.csv')
# a_ada_boost.split_dataset(datasets, 0.7, 'binary')
# a_ada_boost.train()
# a_ada_boost.predict()
