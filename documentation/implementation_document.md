# Implementation document

The core algorithm has three main components: the Tensor class, the FFNN class and the Layer class. The tensor class is used for all of the computations in the neural netork, and it tracks the gradients and other relevant data for the backpropagation algorithm to work. The FFNN class implements a feedforward neural network that uses the Layer class for the weights and biases.

The backpropagation algorithm is called using the backward method from the FFNN class. Training and inference use the forward method from the FFNN class that calls the forward method of each layer in order from the first to last. The training loop is implemented in the model.py file and that can be called by running train.py with the hyperparameters. The trained models can be saved into directories with the model parameters and relevant metadata, and saved models can be evaluated by running evaluate.py.


## Use of large language models

I have used ChatGPT's GPT-5 model to check whether my math has been correct before starting to translate the math into code. I have also used it to study the feedforward neural network architecture.


## List of sources used:



https://michaelkosmider.github.io/dlminiboxtutorial/

https://jaykmody.com/blog/stable-softmax/