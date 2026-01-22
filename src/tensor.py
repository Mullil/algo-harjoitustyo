import numpy as np

class Tensor:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.children = []
        self.grad = 0
        self.backward_fn = None

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
        result.reshape(1, -1)
        result = Tensor(result)
        result.children.append(self)

        def backward_fn(parent):
            children_grad = parent.data.copy()
            children_grad[children_grad < 0] = 0
            children_grad[children_grad > 0] = 1
            parent.children[0].grad = children_grad

        result.backward_fn = backward_fn
        return result

    
