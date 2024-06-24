import numpy as np
from numpy import random

# sigma = 1/(1+exp(-z))
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def add_list(a, b):
    return [x+y for x, y in zip(a, b)]

# 下面的代码，把 (2, 1) 的矩阵变成 (3, 1) 的矩阵，输出是 (3, 1) 的矩阵
# w, a = np.random.randn(k,j), np.random.randn(j,1)
# z = np.dot(w, a)
# 上面的式子等价于 z_k = \sum_j(w_{k,j} a_j) 这里：j = 2, k = 3
#
# In [3]: w, a = np.random.randn(3,2), np.random.randn(2,1)
#
# In [4]: z = np.dot(w, a)
#
# In [5]: z
# Out[5]:
# array([[-1.82356869],
#        [ 0.24324118],
#        [ 0.54224254]])
#
# In [6]:

class Network():
    def __init__(self, topology=(3,2,1)):
        self._topology = topology
        self._build_network(topology)
        self._size = len(topology)

    def forward(self, a):
        for b, w in zip(self._bias, self._weight):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def _build_network(self, topology):
        self._bias = [np.random.randn(i, 1) for i in topology[1:]]
        self._weight = [np.random.randn(k, j) for j, k in zip(topology[:-1], topology[1:])]

    def _backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self._bias]
        nabla_w = [np.zeros(w.shape) for w in self._weight]
        activations = [x]
        zs = []
        activation = x
        for b, w in zip(self._bias, self._weight):
            z = np.dot(w, activation)+b
            activation = sigmoid(z)
            activations.append(activation)
            zs.append(z)

        delta = (activation - y)*sigmoid_prime(zs[-1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta
        for i in range(2, self._size):
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self._weight[-i+1].transpose(), delta)*sp
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
        return (nabla_b, nabla_w)

    def _update_mini_batch(self, train_data, eta):
        nabla_b = [np.zeros(b.shape) for b in self._bias]
        nabla_w = [np.zeros(w.shape) for w in self._weight]
        for a, y in train_data:
            delta_nabla_b, delta_nabla_w = self._backprop(a, y)
            nabla_b = [b+db for b, db in zip(nabla_b, delta_nabla_b)]
            nabla_w = [w+dw for w, dw in zip(nabla_w, delta_nabla_w)]
        self._weight = [w - eta*nw/len(train_data) for w, nw in zip(self._weight, nabla_w)]
        self._bias = [b - eta*nb/len(train_data) for b, nb in zip(self._bias, nabla_b)]

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        data_len = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            for i in range(0, data_len, mini_batch_size):
                self._update_mini_batch(training_data[i:i+mini_batch_size], eta)
            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)}: {len(test_data)}")
            else:
                print("Epoch {epoch} complete")

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for (x, y) in test_results)

Network()
