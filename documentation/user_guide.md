# User guide

The model can be trained by running the train.py script with command line parameters.

An example command shown below trains the feedforward neural network with 2 hidden layers, 32 neurons in each layer for 10 epochs with a learning rate of 0.01 and batch size of 32, and saves the model to a directory called testmodel. 

```{bash}
poetry run python3 train.py --layers 2 --hidden_dim 32 --epochs 10 --lr 0.01 --batc
h_size 32 --model_dir testmodel
```

Runnin train.py script automatically evaluates the performance of the model on test data, but saved models can also be evaluated later by running the evaluate.py script and giving it the saved model directory as a command line parameter. An example is shown below:

```{bash}
poetry run python3 evaluate.py --model_dir testmodel
```

