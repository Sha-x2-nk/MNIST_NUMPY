from jax import numpy as jnp

from modules.kernels.JAXKernels import kernelAdamStep

class Adam:
    def __init__(self, learning_rate = 1e-3, 
                 beta1 = 0.9, beta2 = 0.999, m = 0, 
                 v = 0, t = 0, epsilon = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = m
        self.v = v
        self.t = t
        self.epsilon = epsilon

    def step(self, param: jnp.ndarray, grads: jnp.ndarray):
        # self.t = self.t + 1

        # self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        # mt = self.m / (1 - self.beta1**self.t) #bias correction
        
        # self.v = self.beta2 * self.v + (1 - self.beta2) * (grads * grads)
        # vt = self.v / (1 - self.beta2**self.t) #bias correction

        # param -= self.learning_rate * mt / (jnp.sqrt(vt) + self.epsilon)

        self.m, self.v, self.t, param = kernelAdamStep(param, grads, self.learning_rate, self.beta1, self.beta2, self.m, self.v, self.t, self.epsilon)

        return param
