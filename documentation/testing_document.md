# Testing document

![Coverage report](coverage.png "Coverage report")

Thus far only the tensor class has been tested with unit tests. The tensor class was tested to make sure the tensor operations used for the network work correctly and the gradients are computed correctly. The inputs for the tests were hypothetical very small-scale input vectors, weight matrices and biases.

The tests can done by running

```{bash}
poetry run pytest
```

on the root of the project,
and the coverage report can be seen by running 
```{bash}
coverage run --branch -m pytest
```

and 

```{bash}
coverage report -m
```

on the root of the project.