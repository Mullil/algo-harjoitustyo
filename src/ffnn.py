from typing import List
import numpy as np
from tensor import Tensor
from utils import topological_sort


class Layer:
    """
    Class for a hidden layer in a feedforward neural network
    """

    def __init__(self, in_d, out_d, is_output=False):
        """
        Initializes the layer's weights and biases according to He initialization

        Args:
            in_d: input dimension for the layer
            out_d: output dimension from the layer
            is_output: inidcates whether the layer is the output layer or not
        """
        self.weights = Tensor(np.random.normal(
            0.0, np.sqrt(2/in_d), (out_d, in_d)))
        self.biases = Tensor(np.zeros((out_d, 1)))
        self.is_output = is_output

    def forward(self, x: Tensor, y: Tensor, y_i: int):
        """
        Runs the forward pass on the input. The activation function is ReLU for all layers except
        the output layer. The output layer uses softmax which is used to compute
        the cross-entropy loss.

        Args:
            x: The input tensor
            y: The one-hot-encoded label tensor
            y_i: the correct label

        Returns:
            activation tensor if the layer is not the output layer
            (loss, prediction) if the layer is the output layer
        """
        z = x.preactivation(self.weights, self.biases)
        if not self.is_output:
            a = z.relu()
            return a
        loss = z.cross_entropy_with_softmax(y, y_i)
        return loss, np.argmax(z.softmax(z.data))


class FFNN:
    """
    Class for a feedforward neural network used for image classification
    """

    def __init__(self, layers: List[Layer], lr):
        self.layers = layers
        self.lr = lr
        self.loss = None
        self.prediction = None

    def forward(self, x: Tensor, y: Tensor, y_i: int):
        """
        Calls the forward pass on each layer, where the output of the previous layer becomes the
        input for the next layer. The prediction for the forward pass is
        the index of the largest value of the softmax activation.
        Sets the loss attribute based on the cross-entropy loss.

        Args:
            x: The input tensor
            y: The one-hot-encoded label tensor
            y_i: the correct label
        """
        for layer in self.layers:
            output = layer.forward(x, y, y_i)
            x = output
        if self.loss is not None:
            self.loss += x[0]
        else:
            self.loss = x[0]
        self.prediction = x[1]

    def backward(self):
        """
        Traverses the sorted Directed Acyclic Graphs from the end to the beginning while calling the
        backward function on each node, which computes the gradients w.r.t the biases and weights.
        After computing the gradients, the parameters of each layer are updated by subtracting the
        gradients multiplied by the learning rate.
        """
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
        """
        Re-initializes the gradients of the parameters at each layer to zero
        """
        for layer in self.layers:
            layer.weights.grad = np.zeros(layer.weights.data.shape)
            layer.biases.grad = np.zeros(layer.biases.data.shape)
        self.loss = None
