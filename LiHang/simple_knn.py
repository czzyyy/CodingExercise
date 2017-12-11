# encoding=utf-8
# 参考：http://blog.csdn.net/wds2006sdo/article/details/51933044
import numpy as np
import random
import csv


class Simplest_KNN(object):
    def __init__(self, k, class_num):
        self.k = k
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

    def test(self):
        predict = []
        result = 0
        for j in range(len(self.test_y)):
            knn_list = []       # 当前k个最近邻居
            max_index = -1      # 当前k个最近邻居中距离最远点的坐标
            max_dist = 0        # 当前k个最近邻居中距离最远点的距离

            # 先将前k个点放入k个最近邻居中，填充满knn_list
            for i in range(self.k):
                label = self.train_y[i]
                train_vec = self.train_x[i]

                dist = np.linalg.norm(train_vec - self.test_x[j])         # 计算两个点的欧氏距离

                knn_list.append((dist, label))

            # 剩下的点
            for i in range(self.k, len(self.train_y)):
                label = self.train_y[i]
                train_vec = self.train_x[i]

                dist = np.linalg.norm(train_vec - self.test_x[j])         # 计算两个点的欧氏距离

                # 寻找10个邻近点钟距离最远的点
                if max_index < 0:
                    for k in range(self.k):
                        if max_dist < knn_list[k][0]:
                            max_index = k
                            max_dist = knn_list[max_index][0]

                # 如果当前k个最近邻居中存在点距离比当前点距离远，则替换
                if dist < max_dist:
                    knn_list[max_index] = (dist, label)
                    max_index = -1
                    max_dist = 0

            # 统计选票
            class_total = self.class_num
            class_count = [0 for i in range(class_total)]
            for dist, label in knn_list:
                class_count[int(label)] += 1

            # 找出最大选票
            mmax = max(class_count)

            # 找出最大选票标签
            pred = 0
            for i in range(class_total):
                if mmax == class_count[i]:
                    pred = i
                    predict.append(pred)
                    break
            if int(pred) == int(self.test_y[j]):
                result = result + 1
            if j % 100 == 0:
                print('iter:', j)
        print('acc:', float(result) / float(len(self.test_y)))
        return np.array(predict)
