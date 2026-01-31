# Specification document

The purpose of this project is to create a neural network with a backpropagation algorithm for image classification with Python. Specifically the neural network is trained to classify
handwritten digits in the MNIST dataset. The backpropagation algorithm will be implemented using an autograd system, which will make use of directed acyclic graphs to track the gradients and a topological sorting algorithm to 
sort the directed acyclic graphs for backpropagation. The program will receive the handwritten digits as input with the hyperparameters of the neural network that should be trained.

As described in [this article](https://www.baeldung.com/cs/backpropagation-time-complexity) the time complexity of the backward pass for a rectangular neural network is O(L*N^2), where L is the number of layers and N is the number of neurons in a layer. The space complexity depends on the amount of recomputation that is done during the backward pass. I could not find a source for the space complexity of the backward pass, but it should be the number of weights (L*N^2) plus the number of intermediate values and gradients that are stored, which can differ based on the solution. Thus I would assume a time complexity of O(L*N^2 + V(N+1)), where V is the number of intermediate values and N is the number of neurons, which is also the gradient dimension.

Programming languages I can peer-review projects written in:
Python, JavaScript, C/C++

A source I intend to use:

https://michaelkosmider.github.io/dlminiboxtutorial/


## The core of the project

The core of the project is a backpropagation algorithm that can be used to train a neural network. At first the algorithm will be implemented to suit the training
of a feedforward neural network, but the plan is to make the autograd system flexible enough for different neural network architectures, and if I have time the neural network architecture will be changed to a convolutional neural network, which is more suitable for image data such as the MNIST dataset. The plan is to also make the algorithm flexible for different neural network sizes (e.g. number of layers, hidden dimension, input size).



The study program I belong to is the Bachelor’s program in Computer Science, and the language used in the documentation is English.
