# MNIST_NUMPY

MNIST_NUMPY is a repository containing a python coded implementation of a feed forward neural network using (JAX)NumPy. This implementation consists of a simple 2-layer neural network with 1 hidden layer containing 2048 neurons and an output layer with 10 neurons.

The main purpose of this project was to gain a deeper understanding of backpropagation in neural networks and to implement it from scratch. The code only relies on two libraries: (JAX)NumPy and math.

The code heavily draws inspiration from the assignments of the CS231N course offered by Stanford University. It incorporates various components such as a simple neural network layer, ReLU activation layer, dropout, L2 regularization, and the Adam update rule.

By fine-tuning the dropout probability (pkeep) to an optimal value of 0.7315 using cross-validation and training the model for 20 epochs.

## Preformance
| Device Type | Device Name | Kernels | Time taken (20 epochs) |
|--|--|--|--|
| CPU | Intel i7 12650H | NumPy | 3m 57s |
| CPU | Intel i7 12650H | JAX (CPU) | 47s |
| GPU | Nvidia RTX 3070Ti (Laptop) | JAX (CUDA) | 1.9 s |

## Accuracy
| Dataset | Acc |
| Validation set | 97.8 |
| Train set | 99.21 |
| Test set | 97.7 |

Feel free to explore the code and experiment with different configurations to further enhance the performance of the neural network.
