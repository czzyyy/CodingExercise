import LiHang.kd_tree as kd


class train_data(object):
    def __init__(self, train_x=None, train_y=None):
        self.train_x = train_x
        self.train_y = train_y


class Simple_KNN(object):
    def __init__(self, train_x, train_y, k):
        self.train_x = train_x
        self.train_y = train_y
        self.data_list = self.prepare_data()
        self.sim_kd = None
        self.k = k
        self.train()
        print('train over')

    def prepare_data(self):
        data_list = []
        for i in range(len(self.train_x)):
            data_list.append(train_data(self.train_x[i], self.train_y[i]))
        return data_list

    def train(self):
        self.sim_kd = kd.Simple_KD_Tree(self.data_list, self.k)

    def test(self, test_x, test_y):
        result = 0
        for i in range(len(test_x)):
            x = test_x[i]
            self.sim_kd.find_nn(x)
            vote = {}
            for n in self.sim_kd.k_list:
                if n.element.train_y in vote:
                    vote[n.element.train_y] = vote[n.element.train_y] + 1
                else:
                    vote[n.element.train_y] = 1
            max_n = -1
            pred = 0
            for k in vote.keys():
                if vote[k] > max_n:
                    max_n = vote[k]
                    pred = k
            if int(pred) == int(test_y[i]):
                result = result + 1
        print('acc:', float(result) / float(len(test_y)))


