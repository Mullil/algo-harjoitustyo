import numpy as np


class Tensor:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.children = []
        self.grad = np.zeros(self.data.shape)
        self.backward_fn = lambda x: 0

    def __add__(self, x: "Tensor"):
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
        result = self.data.copy()
        result[result < 0] = 0
        result = Tensor(result)
        result.children.append(self)

        def backward_fn(parent):
            children_grad = parent.data.copy()
            children_grad[children_grad < 0] = 0
            children_grad[children_grad > 0] = 1
            parent.children[0].grad += children_grad

        result.backward_fn = backward_fn
        return result

    def _softmax(self, data):
        result = np.exp(data - np.max(data)) / \
            np.sum(np.exp(data - np.max(data)))
        return result

    def _log_softmax(self, data, y_i):
        result = data[y_i] - np.max(data) - \
            np.log(np.sum(np.exp(data - np.max(data))))
        return result

    def cross_entropy_with_softmax(self, y: "Tensor", y_i: int):
        result = Tensor(-self._log_softmax(self.data, y_i))
        result.children.append(self)

        def backward_fn(parent):
            parent.children[0].grad += self._softmax(
                parent.children[0].data) - y.data

        result.backward_fn = backward_fn
        return result

    def preactivation(self, w: "Tensor", b: "Tensor"):
        result = Tensor(w.data @ self.data + b.data)
        result.children.append(self)
        result.children.append(w)
        result.children.append(b)

        def backward_fn(parent):
            parent.children[1].grad += parent.grad @ parent.children[0].data.T
            parent.children[2].grad += np.ones((parent.children[2].data.shape))

        result.backward_fn = backward_fn
        return result
