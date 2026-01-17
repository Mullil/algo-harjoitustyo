# Weekly report - Week 1

This week I started to plan the implementation of the algorithm and reviewed the structure of a feedforward neural network. For planning the backpropagation algorithm
I did some research on what kind of data structures should be used to track all of the needed values and how an autograd system works. I also started to write the math needed
for the algorithm on paper. At first I made sure I can calculate the gradients of weights on a single layer network and then worked towards a more general solution to compute the
gradients of an arbitrary layer with respect to the weights.

The propgram has not yet really progressed, as this week was spent on reviewing the math and planning the implementation. I have tried writing some smaller parts of the algorithm
to see how things work, but there is not anything to submit yet. This week I learned the structure of a feedforward neural network and the general principles of how an autograd
system would work on a backpropagation algorithm.

Figuring out the math behind the backpropagation algorithm was challenging at first, but it did not take too much time to understand it. It was also challenging to figure out the 
space complexity of the backpropagation algorithm, as I did not find a solution from the internet. However, I am aware that I can affect the space complexity with different choices in
implementation.

Next I will start to implement the autograd system by writing the gradient functions of different operations on vectors/matrices and implementing the graph for the intermediate values and the gradients.
After this I should make the topological sorting algorithm to be able to traverse the graph to compute the gradients.
