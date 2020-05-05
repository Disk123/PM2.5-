import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


train_x=np.load("train_x_24.npy")
train_y=np.load("train_y_24.npy")
test_x=np.load("test_x_24.npy")
test_y = np.load("test_y_24.npy")


# 定义评估函数
# 计算均方误差（Mean Squared Error，MSE）
# r^2 用于度量因变量的变异中 可以由自变量解释部分所占的比例 取值一般为 0~1
#  使用R Square来衡量一个模型的好坏
def r2_score(y_true, y_predict):
    # 计算y_true和y_predict之间的MSE
    MSE = np.sum((y_true - y_predict) ** 2) / len(y_true)
    # 计算y_true和y_predict之间的R Square
    return 1 - MSE / np.var(y_true)

# 线性回归
class LinearRegression:

    def __init__(self):
        # 初始化 Linear Regression 模型
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        # 根据训练数据集X_train, y_train训练Linear Regression模型
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        # 对训练数据集添加 bias
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_gd(self, X_train, y_train, eta=2, n_iters=1e4):
        '''
        :param X_train: 训练集
        :param y_train: label
        :param eta: 学习率
        :param n_iters: 迭代次数
        :return: theta 模型参数
        '''
        # 根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型

        #  这一句用于判断训练样本数据的个数是否与标签数据相同，如果不相同则返回错误
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        # 定义损失函数
        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y) +0.001*np.sum(theta[1:])   #  求平均损失值
            except:
                return float('inf')

        # 对损失函数求导
        def dJ(theta, X_b, y):
            #                        loss
            #  这是矩阵求导公式
            # aa = theta[1:]
            return X_b.T.dot(X_b.dot(theta) - y) / len(y) #+ 0.001*np.sum(theta[1:])

        #  可视化损失函数
        def plot_loss(cur_iter,loss):
            x = np.arange(0, cur_iter, 1)
            plt.plot(x, loss, "y")
            plt.title("loss fuction")
            plt.ylabel("loss")
            plt.xlabel("espoid")
            plt.grid()
            plt.show()

        #   梯度下降法的输入参数  训练数据  标签  权重大小  ada参数 学习率   训练次数   最终模型相差大小（容忍度）
        def gradient_descent(X_b, y, initial_theta,s_grad, eta, n_iters=1e4, epsilon=1e-8):
            '''
            :param X_b: 输入特征向量
            :param y: lebel
            :param initial_theta: 初始参数
            :param eta: 步长
            :param n_iters: 迭代次数
            :param epsilon: 容忍度
            :return:theta：模型参数
            '''
            theta = initial_theta
            cur_iter = 0
            loss = []


            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)     #  对损失函数求导
                R = np.zeros([len(gradient)])      # 正则化处理
                R[1:] = theta[1:]
                gradient = gradient + 0.001*R
                last_theta = theta              #  把当前的权重值赋予现在的last_theta

                # 使用ada方法更改每个权重的学习率
                s_grad += gradient ** 2
                ada = np.sqrt(s_grad)
                theta = theta - eta * gradient/ada      #   更新权值

                loss.append(J(theta, X_b, y))


                #   判断新的权值的损失是否小于前一权值的损失
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                cur_iter += 1
                print("训练次数     %d"%cur_iter)
                if (cur_iter % 1000 == 0 and cur_iter!=0):
                    plot_loss(cur_iter,loss)
            return theta

        #  训练数据，在训练数据中添加bias偏置项
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        #  初始化权重，一开始全部是0

        initial_theta = np.zeros(X_b.shape[1]) # 初始化theta
        s_grad = np.zeros(X_b.shape[1])       #  初始化权重的可变的学习率
        self._theta = gradient_descent(X_b, y_train, initial_theta, s_grad, eta, n_iters)
        #  最终模型的参数输出，第一项为偏置项，其他的为权重大小
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        # 给定待预测数据集X_predict，返回表示X_predict的结果向量
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        #  为预测数据添加一个偏置项
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)         # 返回模型预测的结果

    #  衡量模型的好坏
    def score(self, X_test, y_test):
        # 根据测试数据集 X_test 和 y_test 确定当前模型的准确度
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return

# 模型训练
LR = LinearRegression().fit_gd(train_x, train_y)
# 训练评分
print(LR.score(train_x, train_y))
# 预测评分
print(LR.score(test_x,test_y))
# 预测结果
result = LR.predict(test_x)
print(LR._theta)
# 保存权重
np.save("weight.npy",LR._theta)
print('断点')
