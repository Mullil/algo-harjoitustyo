import numpy as np


class Tensor:
    """
    The class for storing the numerical arrays and the relevant data needed for backpropagation.

    Attributes:
        data: The actual numerical array the tensor consists of
        children: The tensors the current tensor was computed from
        grad: the gradient array for the tensor
        backward_fn: If the tensor was a result of a tensor operation, calling backward_fn
                     computes the gradients of the operands (i.e. the children) w.r.t the loss using
                     the gradient of the result and knowledge of the operation the result was computed with.
    """
    def __init__(self, data: np.ndarray):
        self.data = data
        self.children = []
        self.grad = np.zeros(self.data.shape)
        self.backward_fn = lambda x: 0

    def __add__(self, x: "Tensor"):
        """
        Implements addition for Tensor class
        
        Parameters:
            x: another tensor self is summed with

        Returns:
            The computed result as a Tensor class with a function to compute the gradient of the operation
        """
        result = Tensor(self.data + x.data)
        result.children.append(self)
        result.children.append(x)

        def backward_fn(parent):
            parent.children[0].grad += parent.grad
            parent.children[1].grad += parent.grad

        result.backward_fn = backward_fn
        return result

    def __matmul__(self, x: "Tensor"):
        result = Tensor(self.data @ x.data)
        result.children.append(self)
        result.children.append(x)

        def backward_fn(parent):
            parent.children[0].grad += parent.grad @ parent.children[1].data.T
            parent.children[1].grad += parent.children[0].data.T @ parent.grad

        result.backward_fn = backward_fn
        return result

    def relu(self):
        """
        Implements ReLU activation function
        
        self: an output of a preactivation 
        """
        result = self.data.copy()
        result[result < 0] = 0
        result = Tensor(result)
        result.children.append(self)

        def backward_fn(parent):
            children_grad = parent.data.copy()
            children_grad[children_grad < 0] = 0
            children_grad[children_grad > 0] = 1
            parent.children[0].grad += parent.grad * children_grad

        result.backward_fn = backward_fn
        return result

    def _softmax(self, data):
        result = np.exp(data - np.max(data)) / \
            np.sum(np.exp(data - np.max(data)))
        return result

    def _log_softmax(self, data: np.ndarray, y_i: int):
        """
        Implements a numerically stable version of log_softmax that is used to compute cross-entropy loss

        Parameters:
            data: preactivation from a hidden layer

        Returns:
            The result of the computation as a numpy array
        """
        result = data[y_i] - np.max(data) - \
            np.log(np.sum(np.exp(data - np.max(data))))
        return result

    def cross_entropy_with_softmax(self, y: "Tensor", y_i: int):
        """
        Combines the softmax activation function with cross-entropy loss.
        
        Parameters:
            self: an output of a preactivation function
            y: One-hot-encoded tensor representing the correct class
            y_i: The correct class as an integer

        Returns:
            The computed result as a Tensor class with a function to compute the gradient of the operation
        """
        result = Tensor(-self._log_softmax(self.data, y_i))
        result.children.append(self)

        def backward_fn(parent):
            parent.children[0].grad += self._softmax(
                parent.children[0].data) - y.data

        result.backward_fn = backward_fn
        return result

    def preactivation(self, w: "Tensor", b: "Tensor"):
        """
        Computes the preactivation function of a linear layer and defines the gradients of the operands w.r.t the output
        
        Parameters:
            self: The input for the preactivation
            w: The weight matrix of one layer
            b: The biases of the neurons in one layer
        Returns:
            The computed result as a Tensor class with a function to compute the gradient of the operation
        """
        result = Tensor(w.data @ self.data + b.data)
        result.children.append(self)
        result.children.append(w)
        result.children.append(b)

        def backward_fn(parent):
            parent.children[0].grad = parent.children[1].data.T @ parent.grad
            parent.children[1].grad += parent.grad @ parent.children[0].data.T
            parent.children[2].grad += parent.grad

        result.backward_fn = backward_fn
        return result
