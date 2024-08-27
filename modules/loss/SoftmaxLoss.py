import jax.numpy as jnp

from modules.kernels.JAXKernels import kernelSoftmaxLoss, kernelSoftmaxLossGrad

def softmaxLoss(input_data: jnp.ndarray, labels: jnp.ndarray, mode: str = "train"):
    """
        Computes the loss and gradient for softmax classification.

        Inputs:
        - input_data: Input data, of shape (N, C) where input_data[i, j] is the score for the jth
        class for the ith input.
        - labels: Vector of labels, of shape (N,) where labels[i] is the label for input_data[i] and
        0 <= labels[i] < C

        Returns a tuple of:
        - loss: Scalar giving the loss
        - d_input_data: Gradient of the loss with respect to input_data
    """

    if mode == "train":
        return kernelSoftmaxLossGrad(input_data, labels)

    return kernelSoftmaxLoss(input_data, labels)
