# 参考：http://blog.csdn.net/wds2006sdo/article/details/52849400
import numpy as np
import random
import csv


class Node(object):
    def __init__(self, class_label=None, split=None):
        self.class_label = class_label
        self.split = split
        self.dict = {}

    def add_node(self, feature_value, sub_node):
        self.dict[feature_value] = sub_node


class Decision_Tree(object):
    # type: ID3  C4.5
    def __init__(self, feature_num, threshold, tree_type='ID3'):
        self.tree_type = tree_type
        self.threshold = threshold
        self.feature_num = feature_num
        self.root = None
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
        self.train_x = list([i[1:len(i)] for i in train_set])
        self.train_y = list([i[0] for i in train_set])
        self.test_x = list([i[1:len(i)] for i in copy])
        self.test_y = list([i[0] for i in copy])

    def cal_entry(self, y):
        entry = 0.0
        d = len(y)
        set_y = set(y)
        for label in set_y:
            a = float(len([i for i in y if i == label])) / float(d)
            b = np.log2(a)
            entry -= a * b
        return entry

    def cal_condition_entry(self, fea_x, y):
        con_entry = 0.0
        d = len(y)
        set_x = set(fea_x)
        for fea in set_x:
            fea_y = np.array(y)[fea_x == fea]
            HD_y = self.cal_entry(fea_y)
            con_entry += float(len(fea_y)) / float(d) * HD_y
        return con_entry

    def train(self):
        print('train start')
        self.root = self.recurse_create(self.train_x, self.train_y, [i for i in range(self.feature_num)])
        print('train over')

    def recurse_create(self, train_x, train_y, features):
        label_set = set(train_y)
        if len(label_set) == 1:
            return Node(class_label=label_set.pop())
        if len(features) == 0:
            max_class = 0
            max_nums = 0
            nums = np.zeros(int(max(train_y)))
            for i in train_y:
                nums[int(i)] = nums[int(i)] + 1
            for i in range(len(nums)):
                if nums[i] > max_nums:
                    max_nums = nums[i]
                    max_class = i
            return Node(class_label=max_class)
        else:
            max_feature = 0
            max_gda = 0
            HD = self.cal_entry(train_y)
            for fea in features:
                fea_x = np.array(np.array(train_x)[:, fea].flat)
                HDA = self.cal_condition_entry(fea_x, train_y)
                if self.tree_type == 'ID3':
                    gda = HD - HDA
                elif self.tree_type == 'C4.5':
                    gda = (HD - HDA) / HD
                if gda > max_gda:
                    max_gda = gda
                    max_feature = fea

            print('training:', max_feature)

            if max_gda < self.threshold:
                max_class = 0
                max_nums = 0
                nums = np.zeros(int(max(train_y)) + 1)
                for i in train_y:
                    nums[int(i)] = nums[int(i)] + 1
                for i in range(len(nums)):
                    if nums[i] > max_nums:
                        max_nums = nums[i]
                        max_class = i
                return Node(class_label=max_class)

            left_feature = [x for x in features if x != max_feature]
            node = Node(None, max_feature)
            train_x_feature = np.array(np.array(train_x)[:, max_feature].flat)
            train_x_feature_set = set(train_x_feature)
            for value in train_x_feature_set:
                index = []
                for i in range(len(train_y)):
                    if train_x[i][max_feature] == value:
                        index.append(i)
                sub_train_x = np.array(train_x)[index]
                sub_train_y = np.array(train_y)[index]
                sub_node = self.recurse_create(sub_train_x, sub_train_y, left_feature)
                node.add_node(value, sub_node)

        return node

    def test(self):
        result = 0
        for i in range(len(self.test_y)):
            x = self.test_x[i]
            temp_root = self.root
            while temp_root.class_label is None:
                temp_root = temp_root.dict[x[temp_root.split]]
            pred = temp_root.class_label
            print('pred:', pred, 'label:', self.test_y[i])
            if int(pred) == int(self.test_y[i]):
                result = result + 1
        print('acc:', float(result) / float(len(self.test_y)))

    def predict(self, x):
        temp_root = self.root
        while temp_root.class_label is None:
            temp_root = temp_root.dict[x[temp_root.split]]
        print('predict:', temp_root.class_label)
