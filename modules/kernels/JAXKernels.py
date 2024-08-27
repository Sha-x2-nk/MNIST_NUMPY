import jax
import jax.numpy as jnp

import numpy as np

@jax.jit
def kernelAffineForward(input_data: jnp.ndarray, weight: jnp.ndarray,
                        bias: jnp.ndarray):
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

        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: input_data
    """
    _input_data = input_data.reshape(input_data.shape[0], -1)
    out = _input_data @ weight + bias
    return out

@jax.jit
def kernelAffineBackward(d_out: jnp.ndarray, input_data: jnp.ndarray,
                         weight: jnp.ndarray):
    """
        Computes the backward pass for an affine (fully connected) layer.

        Inputs:
        - d_out: Upstream derivative, of shape (N, M)
        - input_data: Input data, of shape (N, d_1, ... d_k)
        - weight: A numpy array of weights, of shape (D, M)
        - bias: A numpy array of biases, of shape (M,)

        Returns a tuple of:
        - d_input_data: Gradient with respect to input_data, of shape (N, d1, ..., d_k)
        - d_weight: Gradient with respect to weight, of shape (D, M)
        - d_bias: Gradient with respect to bias, of shape (M,)
    """
    d_bias = d_out.sum(axis = 0)

    _input_data = input_data.reshape(input_data.shape[0], -1)

    d_weight = _input_data.T @ d_out
    d_input_data = d_out @ (weight.T)
    d_input_data = d_input_data.reshape(*input_data.shape)

    return d_input_data, d_weight, d_bias

@jax.jit
def kernelReLUForward(input_data: jnp.ndarray):
    """
        Computes the forward pass for a layer of rectified linear units (ReLUs).

        Input:
        - input_data: Inputs, of any shape

        Returns:
        - out: Output, of the same shape as x
    """
    out = jnp.maximum(0, input_data)
    return out

@jax.jit
def kernelReLUBackward(d_out: jnp.ndarray, input_data: jnp.ndarray):
    """
        Computes the backward pass for a layer of rectified linear units (ReLUs).

        Input:
        - d_out: Upstream derivatives, of any shape
        - cache: Input data, of same shape as dout

        Returns:
        - d_input_data: Gradient with respect to input_data
    """
    d_input_data = d_out * (input_data > 0)
    return d_input_data

@jax.jit
def kernelDropoutForward(input_data: jnp.ndarray, p_keep: float):
    """
        Performs the forward pass for (inverted) dropout.

        Inputs:
        - input_data: Input data, of any shape
        - dropout_param: A dictionary with the following keys:
        - p_keep : Dropout parameter. We keep each neuron output with probability p.
        - mode: 'test' or 'train'. If the mode is train, then perform dropout;
                                    If the mo   de is test, then just return the input.

        Outputs:
        - out: Array of the same shape as input_data.
        - cache: mask - is the dropout mask that was used
            to multiply the input; in test mode, mask is None.
    """
    dropout_mask = (jnp.asarray(np.random.rand(*input_data.shape))<p_keep)/p_keep
    out = dropout_mask * input_data
    return (out, dropout_mask)

@jax.jit
def kernelDropoutBackward(d_out: jnp.ndarray, dropout_mask: jnp.ndarray):
    """
        Perform the backward pass for (inverted) dropout.

        Inputs:
        - d_out: Upstream derivatives, of any shape
        - cache: mask from dropout_forward.
    """
    return d_out * dropout_mask

@jax.jit 
def kernelSoftmaxLoss(input_data: jnp.ndarray, labels: jnp.ndarray):
    """
        Computes the loss for softmax classification.

        Inputs:
        - input_data: Input data, of shape (N, C) where input_data[i, j] is the score for the jth
        class for the ith input.
        - labels: Vector of labels, of shape (N,) where labels[i] is the label for input_data[i] and
        0 <= labels[i] < C

        Returns:
        - loss: Scalar giving the loss
    """
    shifted_input_data = input_data - jnp.max(input_data, axis = 1, keepdims=True)
    exp_input_data = jnp.exp(shifted_input_data)

    denom = jnp.sum(exp_input_data, axis = 1, keepdims=True) 
    
    scores = exp_input_data / denom
    scores = scores + 1e-10 # epsilon to prevent -log(0)
    N = input_data.shape[0]
    
    loss = jnp.sum(-jnp.log(scores[jnp.arange(N), labels])) / N

    return loss

@jax.jit
def kernelSoftmaxLossGrad(input_data: jnp.ndarray, labels: jnp.ndarray):
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
    shifted_input_data = input_data - jnp.max(input_data, axis = 1, keepdims=True)
    exp_input_data = jnp.exp(shifted_input_data)

    denom = jnp.sum(exp_input_data,axis = 1, keepdims=True) 
    
    scores = exp_input_data / denom
    scores = scores + 1e-10 # epsilon to prevent -log(0)
    N = input_data.shape[0]
    
    loss = jnp.sum(-jnp.log(scores[jnp.arange(N), labels])) / N

    d_input_data = scores
    d_input_data = d_input_data.at[jnp.arange(N), labels].add(-1)
    d_input_data /= N
    return loss, d_input_data

@jax.jit
def kernelAdamStep(param: jnp.ndarray, grads: jnp.ndarray,
                   learning_rate = 1e-3, beta1 = 0.9, 
                   beta2 = 0.999, m = 0, v = 0, t = 0, 
                   epsilon = 1e-8):
    """
        Standard Adam update rule.
    """
    t = t+1

    m = beta1 * m + (1 - beta1) * grads
    mt = m / (1 - beta1**t) #bias correction
    
    v = beta2 * v + (1 - beta2) * (grads * grads)
    vt = v / (1 - beta2**t) #bias correction

    param -= learning_rate * mt / (jnp.sqrt(vt) + epsilon)
    return (m, v, t, param)