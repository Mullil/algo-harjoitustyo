# Weekly report 4

This week I still had to do slight corrections to my backpropagation algorithm, and after fixing them the network was able to quickly get around 95 % accuracy on the training data after 1 or 2 epochs. I also started to document the code and wrote tests for the neural network to check that all of the weights are changed after each backward pass and that the network is able to overfit to a small subset of the dataset with 100 % accuracy. Also the training loop now works with all batch sizes, although currently the batching only affects the weight updates whereas the forward pass is still done one example at a time.

This week I learned that the neural network can reach  surprisingly high accuracy on a training set with an incorrectly implemented backpropagation algorithm. After that I learned to finally make the backpropagation algorithm work correctly.

I don't think anything was particularly unclear or too difficult. Computing the forward pass on a batch feels a bit difficult for now, but I just need to start planning and visualizing it on paper.

Next I will implement the evaluation loop that computes accuracy on the test dataset, and probably make the batching work for the forward pass. I might also still need to figure out more tests for the algorithm. I'm also planning to make a feature that allows to save trained models into files, but that is not as relevant as the previously mentioned features. Having it for the demo could be useful though.

This week I used around 8 hours for the project.