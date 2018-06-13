# -*- coding: utf-8 -*-
# @Time    : 6/12/2018 4:01 PM
# @Author  : sunyonghai
# @File    : linear_unit.py
# @Software: ZJ_AI

from perceptron import Perceptron

def f(x):
    """
    activate function
    :param x:
    :return:
    """
    return x

class LinearUnit(Perceptron):
    def __init__(self, input_num, activator):
        Perceptron.__init__(self, input_num, activator)

def get_training_dataset():
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels

def train_linear_unit():
    lu = LinearUnit(1, f)
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    return lu

if __name__ == '__main__':
    linear_unit = train_linear_unit()
    # 打印训练获得的权重
    print(linear_unit)
    # 测试
    print('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))
