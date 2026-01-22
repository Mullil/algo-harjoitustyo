import unittest
from src.tensor import Tensor
import numpy as np
import numpy.testing as npt

class TestTensor(unittest.TestCase):
    def test_sum_of_tensors_works(self):
        a = Tensor(np.array([[1,2],
                             [1,2]]))
        b = Tensor(np.array([[2,2],
                             [2,2]]))
        c = a + b
        self.assertEqual(type(c), Tensor)
        self.assertEqual(c.children[0], a)
        self.assertEqual(c.children[1], b)
        npt.assert_array_equal(c.data, np.array([[3, 4],
                                                [3, 4]]))
    def test_matmul_of_tensors_works(self):
        a = Tensor(np.array([[1,3],
                             [1,2]]))
        b = Tensor(np.array([[2,2],
                             [2,1]]))
        c = a @ b
        self.assertEqual(type(c), Tensor)
        self.assertEqual(c.children[0], a)
        self.assertEqual(c.children[1], b)
        npt.assert_array_equal(c.data, np.array([[8, 5],
                                                [6, 4]]))
        
    def test_relu_works(self):
        a = Tensor(np.array([[1,2], [1,2]]))
        relu_a = a.relu()
        self.assertEqual(relu_a.children[0], a)
        npt.assert_array_equal(relu_a.data, a.data)
        b = Tensor(np.array([[-1, 2], [1, -2]]))
        relu_b = b.relu()
        npt.assert_array_equal(relu_b.data, np.array([[0, 2], [1, 0]]))

        # Make sure relu does not change the data of the children
        self.assertRaises(AssertionError, lambda: npt.assert_array_equal(relu_b.data, b.data))

    def test_grad_fn_works(self):
        a = Tensor(np.array([[1, 2]]))

        b = Tensor(np.array([[2, -1]]))

        c = Tensor(np.array([[2, -1],
                            [2, 2]]))
        mul = b @ c # [[2, -4]]
        preactivation = a + mul # [[3, -2]]
        activation = preactivation.relu() #[[3, 0]]
        activation.grad = activation.data
        self.assertEqual(0, preactivation.grad)
        activation.backward_fn(activation)
        npt.assert_array_equal(preactivation.grad, np.array([[1, 0]]))
        preactivation.backward_fn(preactivation)
        npt.assert_array_equal(mul.grad, np.array([[1,0]]))
        npt.assert_array_equal(np.array([[1,0]]), a.grad)
        mul.backward_fn(mul)
        npt.assert_array_equal(b.grad, np.array([[2, 2]]))
        npt.assert_array_equal(c.grad, np.array([[2, 0], [-1, 0]]))
