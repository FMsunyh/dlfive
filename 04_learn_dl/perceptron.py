# -*- coding: utf-8 -*-
# @Time    : 6/12/2018 2:22 PM
# @Author  : sunyonghai
# @File    : perceptron.py
# @Software: ZJ_AI

class Perceptron(object):
    def __init__(self,input_num, activator):
        """

        :param input_num:
        :param activator:
        """
        self.activator = activator

        # set initialize weight 0
        self.weights = [0.0 for _ in range(input_num)]

        self.bias = 0.0

    def __str__(self):
        return 'weights\t:{}\nbias\t:{}\n'.format(self.weights, self.bias)

    def predict(self, input_vec):
        """
        Get the predict result
        :return:
        """
        sum = 0
        for (x, w) in zip(input_vec, self.weights):
            sum += x*w

        res = self.activator(sum + self.bias)
        return res

    def train(self, input_vecs, labels, iteration, rate):
        """

        :param input_vecs:
        :param labels:
        :param iteration:
        :param rate:
        :return:
        """
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        """

        :return:
        """
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        delta = label - output

        for idx ,(x, w) in enumerate(zip(input_vec, self.weights)):
            self.weights[idx] +=  rate*delta*x
        self.bias += rate*delta

def f(x):
    """
    activate function
    :param x:
    :return:
    """
    return 1 if x > 0 else 0

def get_training_dataset():
    """
    input data
    :return:
    """

    input_vecs =[[1,1], [0,0], [1,0], [0,1]]
    labels = [1, 0, 0, 0]
    return input_vecs, labels

def train_and_perceptron():
    p = Perceptron(2, f)
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p

if __name__ == '__main__':
    and_perception = train_and_perceptron()
    print(and_perception)

    print('1 and 1 = %d' % and_perception.predict([1, 1]))
    print('0 and 0 = %d' % and_perception.predict([0, 0]))
    print('1 and 0 = %d' % and_perception.predict([1, 0]))
    print('0 and 1 = %d' % and_perception.predict([0, 1]))

