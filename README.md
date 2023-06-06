MNIST_NUMPY

MNIST_NUMPY is a repository containing a python coded implementation of a feed forward neural network using NumPy. This implementation consists of a simple 2-layer neural network with 1 hidden layer containing 2048 neurons and an output layer with 10 neurons.

The main purpose of this project was to gain a deeper understanding of backpropagation in neural networks and to implement it from scratch. The code only relies on two libraries: NumPy and math.

The code heavily draws inspiration from the assignments of the CS231N course offered by Stanford University. It incorporates various components such as a simple neural network layer, ReLU activation layer, dropout, L2 regularization, and the Adam update rule.

By fine-tuning the dropout probability (pkeep) to an optimal value of 0.7315121964113247 using cross-validation and training the model for 20 epochs (which took approximately 5 minutes on an Intel i5 1135G7 processor), the achieved accuracy results are as follows:

Best Validation Accuracy: 0.98
Training Accuracy: 0.9921259842519685
Test Accuracy: 0.9767
Feel free to explore the code and experiment with different configurations to further enhance the performance of the neural network.
