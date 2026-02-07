import unittest
import numpy.testing as npt
from src.model import Model
from src.ffnn import FFNN, Layer

class TestFFNN(unittest.TestCase):
    def setUp(self):
        """
        Initializes a small neural network and the hyperparameters for training
        """
        self.hyperparameters = {"hidden": 32, "layers": 2, "epochs": 20, "lr": 0.01, "batch_size": 1}
        self.image_tensors, self.label_tensors, self.y_i_array = Model(self.hyperparameters)._create_training_set(test=True)
        model_layers = [Layer(784, self.hyperparameters["hidden"]), Layer(self.hyperparameters["hidden"], 10, is_output=True)]
        self.nn = FFNN(model_layers, lr=self.hyperparameters["lr"])

    def test_weights_are_changed_on_backward_pass(self):
        """
        Runs the forward pass and backward pass for the first 50 instances in the training data.
        The test makes sure that after each backward pass the weights are changed.
        """
        for x, y, y_i in zip(self.image_tensors[:50], self.label_tensors[:50], self.y_i_array[:50]):
            self.nn.forward(x, y, y_i)
            old_weights = [layer.weights.data for layer in self.nn.layers]
            self.nn.backward()
            new_weights = [layer.weights.data for layer in self.nn.layers]
            for old, new in zip(old_weights, new_weights):
                self.assertEqual(old.shape, new.shape)
                self.assertRaises(AssertionError, lambda: npt.assert_array_equal(old, new))
            old_weights = new_weights
            self.nn.zero_grad()

    def test_model_can_reach_perfect_accuracy_on_subset(self):
        """
        Runs for a maximum of 20 epochs and trains on the 50 first instances.
        If the model is not able to get 100% accuracy during training, the test fails.
        """
        for epoch in range(self.hyperparameters["epochs"]):
            correct = 0
            for x, y, y_i in zip(self.image_tensors[:50], self.label_tensors[:50], self.y_i_array[:50]):
                self.nn.forward(x, y, y_i)
                correct += self.nn.prediction == y_i
                self.nn.backward()
                self.nn.zero_grad()
            if correct == 50:
                break
        self.assertEqual(correct, 50)
