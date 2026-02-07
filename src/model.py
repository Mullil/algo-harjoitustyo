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

    def __init__(self, hyperparameters: dict):
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

    def train_model(self):
        """
        The training loop that creates and trains the model according to the hyperparameters.
        Reports the average training loss of each epoch and the accuracy of the epoch
        """
        if self.layers == 1:
            model_layers = [Layer(784, 10, is_output=True)]
        else:
            model_layers = [Layer(784, self.hidden)
                            for _ in range(self.layers - 1)]
            model_layers.append(Layer(self.hidden, 10, is_output=True))
        nn = FFNN(model_layers, lr=self.lr)

        image_tensors, label_tensors, y_i_array = self._create_training_set()
        print("\nTraining started")
        for epoch in range(self.epochs):
            correct = 0
            losses = []
            for i in range(0, len(image_tensors), self.batch_size):
                for j in range(i, min(i+self.batch_size, len(image_tensors))):
                    nn.forward(image_tensors[j],
                               label_tensors[j], y_i_array[j])
                    correct += nn.prediction == y_i_array[j]
                losses.append(nn.loss.data / self.batch_size)
                nn.backward()
                nn.zero_grad()
            print(
                f"\nEpoch {epoch} avg training loss: {np.mean(np.array(losses))}")
            print(
                f"Epoch {epoch} training accuracy: {correct / len(image_tensors)}")
