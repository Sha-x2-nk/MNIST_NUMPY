from jax import numpy as jnp
import numpy as np

from modules.kernels.JAXKernels import kernelAffineForward, kernelAffineBackward

class AffineLayer:
    def __init__(self, in_features, out_features):
        self.params = {
            'weight': jnp.asarray(np.random.randn(in_features, out_features)) * jnp.sqrt(2/in_features),
            'bias': jnp.zeros(out_features)
        }

        self.grads = {'weight': 0, 'bias': 0}
        self.cache = None

    def __call__(self, input_data, mode = "train"):
        return self.forward(input_data, mode)
    
    def forward(self, input_data, mode = "train"):
        """
            Computes the forward pass for an affine (fully connected) layer.

            The input_data has shape (N, d_1, ..., d_k) and contains a minibatch of N
            examples, where each example x[i] has shape (d_1, ..., d_k). We will
            reshape each input into a vector of dimension D = d_1 * ... * d_k, and
            then transform it to an output vector of dimension M.

            Inputs:
            - input_data: A numpy array containing input data, of shape (N, d_1, ..., d_k)
            - weight: A numpy array of weights, of shape (D, M)
            - bias: A numpy array of biases, of shape (M,)

            Returns and stores:
            - out: output, of shape (N, M)
            - cache: input_data
        """
        out = kernelAffineForward(input_data, self.params['weight'], 
                                  self.params['bias'])
        if mode == "train":
            self.cache = input_data

        return out

    def backward(self, d_out):
        """
            Computes the backward pass for an affine (fully connected) layer.

            Inputs:
            - d_out: Upstream derivative, of shape (N, M)
            - cache: x: Input data, of shape (N, d_1, ... d_k)

            Returns and stores:
            - d_input_data: Gradient with respect to input_data, of shape (N, d1, ..., d_k)
            - d_weight: Gradient with respect to weight, of shape (D, M)
            - d_bias: Gradient with respect to bias, of shape (M,)
        """
        input_data = self.cache

        d_input_data, d_weight, d_bias = kernelAffineBackward(d_out,
                                                              input_data,
                                                              self.params['weight'])
        self.grads['weight'] = d_weight
        self.grads['bias'] = d_bias

        return d_input_data
