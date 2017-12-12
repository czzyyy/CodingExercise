import numpy as np
import pandas as pd
import math
import copy
import matplotlib.pyplot as plt


# 2-D data
class Mix_Gaussians(object):
    def __init__(self, k, iter_num):
        self.k = k
        self.iter_num = iter_num
        self.X = None
        self.r = None
        self.sigma = np.random.random([k, 2, 2])
        self.mu = np.random.random([k, 2])
        self.alpha = np.array([1.0 / k] * k)

    def generate_data(self, N, mus, sigmas):
        self.X = np.zeros([N, 2])
        self.r = np.zeros([N, self.k])
        # for i in range(len(mus)):
        #     self.mu[i][0] = mus[i][0]
        #     self.mu[i][1] = mus[i][1]
        #     self.sigma[i][0][0] = sigmas[i][0][0]
        #     self.sigma[i][0][1] = sigmas[i][0][1]
        #     self.sigma[i][1][0] = sigmas[i][1][0]
        #     self.sigma[i][1][1] = sigmas[i][1][1]
        self.mu = np.array(mus)
        self.sigma = np.array(sigmas)
        for i in range(N):
            ran = np.random.randint(0, self.k)
            self.X[i, :] = np.random.multivariate_normal(self.mu[ran], self.sigma[ran], 1)

    def load_data_from_csv(self, filename):
        raw_data = pd.read_csv(filename, header=0)
        self.X = raw_data.values
        self.r = np.zeros([len(self.X), self.k])

    def e_step(self):
        N = len(self.X)
        for i in range(N):
            denom = 0.0
            for j in range(self.k):
                sigma = np.matrix(self.sigma[j])
                denom += self.alpha[j] * math.exp(-0.5 * (np.array(
                    self.X[i, :] - self.mu[j, :]).dot(sigma.I) * np.transpose(
                    [self.X[i, :] - self.mu[j, :]]))) / np.sqrt(np.linalg.det(sigma)*4*np.pi*np.pi)
            for j in range(self.k):
                sigma = np.matrix(self.sigma[j])
                numer = self.alpha[j] * math.exp(-0.5 * (np.array(
                    self.X[i, :] - self.mu[j, :]).dot(sigma.I) * np.transpose(
                    [self.X[i, :] - self.mu[j, :]]))) / np.sqrt(np.linalg.det(sigma)*4*np.pi*np.pi)
                self.r[i, j] = numer / denom

    def m_step(self):
        N = len(self.X)
        for i in range(self.k):
            sum_r = 0.0
            mu_numer = 0.0
            sigma_numer = 0.0
            for j in range(N):
                sum_r += self.r[j, i]
                mu_numer += self.r[j, i] * self.X[j, :]
                sigma_numer += self.r[j, i] * np.transpose([self.X[j, :] - self.mu[i, :]]
                                                           ).dot([self.X[j, :] - self.mu[i, :]])
            self.mu[i, :] = mu_numer / sum_r
            self.sigma[i, :] = sigma_numer / sum_r
            self.alpha[i] = sum_r / N

    def train(self):
        for i in range(self.iter_num):
            err = 0.0
            err_alpha = 0.0
            old_mu = copy.deepcopy(self.mu)
            old_alpha = copy.deepcopy(self.alpha)
            self.e_step()
            self.m_step()
            print("迭代次数:", i+1)
            print("估计的均值:", self.mu)
            print("估计的混合项系数:", self.alpha)
            for z in range(self.k):
                err += (abs(old_mu[z, 0] - self.mu[z, 0]) + abs(old_mu[z, 1] - self.mu[z, 1]))
                err_alpha += (abs(old_alpha[z] - self.alpha[z]))
            if (err < 0.00001) and (err_alpha < 0.00001):
                print(err, err_alpha)
                break
        print("估计的均值:", self.mu)
        print("估计的协方差矩阵:", self.sigma)
        print("估计的混合项系数:", self.alpha)

        # draw
        probility = np.zeros(len(self.X))
        plt.subplot(221)
        plt.scatter(self.X[:, 0], self.X[:, 1], c='b', s=25, alpha=0.4, marker='o')
        plt.title('random generated data')
        plt.subplot(222)
        plt.title('classified data through EM')
        order = np.zeros(len(self.X))
        color = ['b', 'r', 'k', 'y', 'violet', 'slategray', 'turquoise', 'fuchsia', 'goldenrod', 'cyan', 'deeppink',
                 'beige', 'blueviolet', 'darkkhaki', 'moccasin']
        for i in range(len(self.X)):
            for j in range(self.k):
                if self.r[i, j] == max(self.r[i, :]):
                    order[i] = j
                sigma = np.matrix(self.sigma[j])
                probility[i] += self.alpha[int(order[i])]*math.exp(-0.5 * (np.array(
                    self.X[i, :] - self.mu[j, :]).dot(sigma.I) * np.transpose(
                    [self.X[i, :] - self.mu[j, :]]))) / np.sqrt(np.linalg.det(sigma)*4*np.pi*np.pi)
            plt.scatter(self.X[i, 0], self.X[i, 1], c=color[int(order[i])], s=25, alpha=0.4, marker='o')
        ax = plt.subplot(223, projection='3d')
        plt.title('3d view')
        for i in range(len(self.X)):
            ax.scatter(self.X[i, 0], self.X[i, 1], probility[i], c=color[int(order[i])])
        plt.show()

    def test(self, test_x):
        denom = 0.0
        r = np.zeros([1, self.k])
        pred = 0
        for j in range(self.k):
            sigma = np.matrix(self.sigma[j])
            denom += self.alpha[j] * math.exp(-0.5*np.array(test_x - self.mu[j, :]).dot(sigma.I) * np.transpose(
                [test_x - self.mu[j, :]])) / np.sqrt(np.linalg.det(sigma)*4*np.pi*np.pi)
        for j in range(self.k):
            sigma = np.matrix(self.sigma[j])
            numer = self.alpha[j] * math.exp(-0.5*np.array(test_x - self.mu[j, :]).dot(sigma.I) * np.transpose(
                [test_x - self.mu[j, :]])) / np.sqrt(np.linalg.det(sigma)*4*np.pi*np.pi)
            r[1, j] = numer / denom
        for i in range(self.k):
            if r[1, i] == max(r[1, :]):
                pred = i
                print("服从的高斯分布的均值为:", self.mu[int(pred), :])
                print("服从的高斯分布的协方差为:", self.sigma[int(pred)])
                break

#简单使用
# import LiHang.mixtures_gaussians as mg
# mus = [
#     [5, 35],
#     [30, 40],
#     [20, 20],
#     [45, 15],
#     [25, 7],
#     [2, 17]
# ]
# sigmas = [
#     [[30, 0], [2, 30]],
#     [[30, 14], [0, 30]],
#     [[30, 3], [17, 30]],
#     [[30, 2], [6, 30]],
#     [[30, 0], [0, 30]],
#     [[30, 22], [16, 30]]
# ]
# a_mix_gaussians = mg.Mix_Gaussians(6, 1000)
# a_mix_gaussians.generate_data(1000, mus=mus, sigmas=sigmas)
# a_mix_gaussians.train()
