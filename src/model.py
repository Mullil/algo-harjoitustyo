import os
import json
import numpy as np
from mnist import MNIST
from tensor import Tensor
from ffnn import FFNN, Layer


class Model:
    """
    Class for initializing the neural network and training it.

    Attributes:
        lr: learning rate used for stochastic gradient descent
        layers: number of hidden layers in the network
        epochs: number of times the training loop will iterate over the whole training set
        hidden: hidden dimension of the layers, i.e. the number of neurons per layer
        batch_size: number of training instances the forward pass is run on before
        calling backward pass on their summed loss
    """

    def __init__(self, hyperparameters, load_existing=False):
        """
        Initializes the hyperparameters for training, if a model is trained.
        
        Args:
            hyperparameters: dictionary of model hyperparameters or None, if load_existing is true
            load_existing: boolean indicating if the class is initialized using a trained model
        """
        if not load_existing:
            self.lr = hyperparameters["lr"]
            self.layers = hyperparameters["layers"]
            self.epochs = hyperparameters["epochs"]
            self.hidden = hyperparameters["hidden"]
            self.batch_size = hyperparameters["batch_size"]

    def _create_training_set(self, test=False):
        """
        Transforms the MNIST training data into tensors the model can use directly.

        Args:
            test: boolean variable to indicate whether the code is called by tests or not

        Returns:
            image_tensors: the images as Tensor objects
            label_tensors: one-hot-encoded tensors
            y_i_array: list with the correct labels of each image
        """
        print("Creating training set")
        mndata = MNIST('../MNIST_data/') if not test else MNIST('MNIST_data/')
        images, labels = mndata.load_training()
        image_tensors = [Tensor(np.array([image]).T / 255) for image in images]
        label_tensors = []
        y_i_array = []
        for label in labels:
            one_hot = np.zeros((10, 1))
            one_hot[label] = 1
            label_tensors.append(Tensor(one_hot))
            y_i_array.append(label)
        print("\nDone creating training set")
        return image_tensors, label_tensors, y_i_array

    def _create_test_set(self):
        """
        Transforms the MNIST test data into tensors the model can use directly.

        Returns:
            image_tensors: the images as Tensor objects
            label_tensors: one-hot-encoded tensors
            y_i_array: list with the correct labels of each image
        """
        print("Creating test set")
        self.mndata = MNIST('../MNIST_data/')
        self.images, labels = self.mndata.load_testing()
        image_tensors = [Tensor(np.array([image]).T / 255) for image in self.images]
        label_tensors = []
        y_i_array = []
        for label in labels:
            one_hot = np.zeros((10, 1))
            one_hot[label] = 1
            label_tensors.append(Tensor(one_hot))
            y_i_array.append(label)
        print("\nDone creating test set")
        return image_tensors, label_tensors, y_i_array

    def train_model(self, model_dir):
        """
        The training loop that creates and trains the model according to the hyperparameters.
        Reports the average training loss of each epoch and the accuracy of the epoch
        """
        if self.layers == 1:
            model_layers = [Layer(784, 10, is_output=True)]
        else:
            model_layers = []
            model_layers.append(Layer(784, self.hidden))
            for _ in range(self.layers - 2):
                model_layers.append(Layer(self.hidden, self.hidden))

            model_layers.append(Layer(self.hidden, 10, is_output=True))
        self.nn = FFNN(model_layers, lr=self.lr)

        image_tensors, label_tensors, y_i_array = self._create_training_set()
        print("\nTraining started")
        for epoch in range(self.epochs):
            correct = 0
            losses = []
            for i in range(0, len(image_tensors), self.batch_size):
                for j in range(i, min(i+self.batch_size, len(image_tensors))):
                    self.nn.forward(image_tensors[j],
                               label_tensors[j], y_i_array[j])
                    correct += self.nn.prediction == y_i_array[j]
                losses.append(self.nn.loss.data / self.batch_size)
                self.nn.backward()
                self.nn.zero_grad()
            print(
                f"\nEpoch {epoch} avg training loss: {np.mean(np.array(losses))}")
            print(
                f"Epoch {epoch} training accuracy: {correct / len(image_tensors)}")

        if model_dir:
            print(f"\nSaved model to directory {model_dir}")
            self.save_model(model_dir)


    def test_model(self):
        """
        The testing loop that calls forward passes on the test data and computes the test accuracy
        """
        image_tensors, label_tensors, y_i_array = self._create_test_set()
        correct = 0
        for i in range(0, len(image_tensors)):
            self.nn.forward(image_tensors[i],
                            label_tensors[i], y_i_array[i])
            correct += self.nn.prediction == y_i_array[i]
            if self.nn.prediction != y_i_array[i]:
                print(self.mndata.display(self.images[i]))
                print(self.nn.prediction)
                print(y_i_array[i])
        print(f"Test accuracy: {correct / len(image_tensors)}")

    def save_model(self, dir):
        """
        Saves the model parameters with relevant metadata into a directory
        to be able to reuse a trained model
        """
        os.mkdir(dir)
        layer_dict = {}
        for i, layer in enumerate(self.nn.layers):
            layer_dict[f"layer{i}_weights"] = layer.weights.data
            layer_dict[f"layer{i}_biases"] = layer.biases.data
        np.savez(f"{dir}/layers.npz", **layer_dict)

        model_metadata = {"layers": self.layers, "hidden": self.hidden}
        with open(f"{dir}/metadata.json", "w") as f:
            json.dump(model_metadata, f)

    def load_model(self, dir):
        """
        Loads a trained model and initializes self.nn with the loaded parameters
        
        Args:
            dir: directory where the model parameters and metadata were saved
        """
        layers = np.load(f"{dir}/layers.npz")
        with open(f"{dir}/metadata.json", "r") as f:
            metadata = json.load(f)
        num_layers = metadata["layers"]
        hidden = metadata["hidden"]
        if num_layers == 1:
            model_layers = [Layer(784, 10, is_output=True)]
        else:
            model_layers = [Layer(784, hidden)
                            for _ in range(num_layers - 1)]
            model_layers.append(Layer(hidden, 10, is_output=True))

        for i in range(num_layers):
            model_layers[i].weights.data = layers[f"layer{i}_weights"]
            model_layers[i].biases.data = layers[f"layer{i}_biases"]

        self.nn = FFNN(model_layers, lr=None)
