from tensor import Tensor
from typing import List
from utils import topological_sort
import numpy as np


class Layer:
    def __init__(self, in_d, out_d, is_output=False):
        self.weights = Tensor(np.random.rand(out_d, in_d))
        self.biases = Tensor(np.random.rand(out_d, 1))
        self.is_output = is_output

    def forward(self, x: Tensor, y: Tensor, y_i: int):
        z = x.preactivation(self.weights, self.biases)
        if not self.is_output:
            a = z.relu()
        elif self.is_output:
            a = z.cross_entropy_with_softmax(y, y_i)
        return a


class FFNN:
    def __init__(self, layers: List[Layer], lr):
        self.layers = layers
        self.lr = lr
        self.loss = 0
        self.prediction = None

    def forward(self, input: Tensor, y: Tensor, y_i: int):
        for layer in self.layers:
            output = layer.forward(input, y, y_i)
            input = output
        self.loss = input
        self.prediction = np.argmax(self.loss.children[0].data)

    def backward(self):
        sorted_nodes: List[Tensor] = topological_sort(self.loss)
        sorted_nodes[-1].grad = sorted_nodes[-1].data
        for i in range(len(sorted_nodes) - 1, -1, -1):
            sorted_nodes[i].backward_fn(sorted_nodes[i])
        for layer in self.layers:
            layer.weights = Tensor(
                layer.weights.data - self.lr * layer.weights.grad)
            layer.biases = Tensor(layer.biases.data -
                                  self.lr * layer.biases.grad)

    def zero_grad(self):
        for layer in self.layers:
            layer.weights.grad = np.zeros(layer.weights.data.shape)
            layer.biases.grad = np.zeros(layer.biases.data.shape)
