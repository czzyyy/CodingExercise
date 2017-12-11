import numpy as np


class Simple_KD_Node(object):
    def __init__(self, element=None, split=None, dist=None, left=None, right=None):
        self.element = element
        self.split = split
        self.dist = dist
        self.left = left
        self.right = right


class Simple_KD_Tree(object):
    # train_list has element train_data, class train_data has
    # two element train_x:[2, 3, 4, 1, 7]  and  train_y:[1] or [2]
    def __init__(self, train_list, k):
        self.train_list = train_list
        self.root = self.create(self.train_list)
        self.k_list = []
        self.k = k
        print('create over')

    def create(self, dataList):
        LEN = len(dataList)
        if LEN == 0:
            return
        dimension = len(dataList[0].train_x)
        max_var = 0
        split = 0
        for i in range(dimension):
            l = []
            for t in dataList:
                l.append(t.train_x[i])
            var = self.compute_var(l)
            if var > max_var:
                max_var = var
                split = i
        dataList.sort(key=lambda x: x.train_x[split])
        point = dataList[int(LEN/2)]
        root = Simple_KD_Node(point, split)
        root.left = self.create(dataList[0:int(LEN/2)])
        root.right = self.create(dataList[int(LEN/2) + 1:LEN])
        return root

    def compute_var(self, arrayList):
        for ele in arrayList:
            ele = float(ele)
        var = np.var(np.array(arrayList))
        return var

    def compute_dist(self, pt1, pt2):
        pt1 = np.array(pt1)
        pt2 = np.array(pt2)
        return np.sqrt(np.sum((pt1-pt2) * (pt1-pt2)))

    # query is test data : [1, 2, 3, 4, 5...]
    def find_nn(self, query):
        node_list = []
        temp_root = self.root
        while temp_root:
            node_list.append(temp_root)
            cur_split = temp_root.split
            if query[cur_split] <= temp_root.element.train_x[cur_split]:
                temp_root = temp_root.left
            else:
                temp_root = temp_root.right

        # look back
        back_point = node_list.pop()
        min_dist = self.compute_dist(back_point.element.train_x, query)
        back_point.dist = min_dist
        self.k_list.append(back_point)
        nn = back_point
        k_set = set(self.k_list)
        while node_list:
            back_point = node_list.pop()
            cur_dist = self.compute_dist(back_point.element.train_x, query)
            back_point.dist = cur_dist
            if cur_dist < min_dist:
                min_dist = cur_dist
                k_set.add(back_point)
                nn = back_point
            elif len(k_set) < self.k:
                k_set.add(back_point)
            cur_split = back_point.split
            if abs(query[cur_split] - back_point.element.train_x[cur_split]) < min_dist:
                if query[cur_split] <= back_point.element.train_x[cur_split]:
                    temp_root = back_point.right
                else:
                    temp_root = back_point.left

                if temp_root:
                    cur_dist = self.compute_dist(temp_root.element.train_x, query)
                    temp_root.dist = cur_dist
                    node_list.append(temp_root)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        k_set.add(temp_root)
                        nn = temp_root
                    elif len(k_set) < self.k:
                        k_set.add(temp_root)
        self.k_list = list(k_set)
        self.k_list.sort(key=lambda x: x.dist)
        return nn, min_dist


