from jax import numpy as jnp

from modules.layers.AffineLayer import AffineLayer
from modules.layers.AffineReLULayer import AffineReLULayer

from modules.loss.SoftmaxLoss import softmaxLoss

from modules.optimisers.Adam import Adam

class NN:
    def __init__(self, p_keep = 1, reg = 0):
        self.layers = {
            'l1': AffineReLULayer(784, 2048, p_keep = p_keep),
            'l2': AffineLayer(2048,10)
        }
        
        self.mode = "eval"
         
        self.adam_configs = {}
        
        for layer in self.layers.keys():
            self.adam_configs[layer] = {}
            for param in self.layers[layer].params:
                self.adam_configs[layer][param] = Adam(learning_rate = 1e-3, beta1 = 0.9, 
                                                       beta2 = 0.999, m = 0, v = 0, 
                                                       t = 0, epsilon = 1e-8)
        
        self.reg = reg
        
    def train(self):
        self.mode = "train"
    def eval(self):
        self.mode = "eval"

    def __call__(self, input_data, labels =None):
        return self.forward(input_data, labels)

    def forward(self, input_data, labels= None):
        out = input_data

        for layer in self.layers.keys():
            out = self.layers[layer].forward(out, mode = self.mode)

        if labels is None:
            return out
        
        if self.mode == "eval":
            loss = softmaxLoss(out, labels, self.mode)
            return out, loss
        
        loss, dout = softmaxLoss(out, labels, self.mode)
        
        if self.reg > 0:
            for layer in self.layers.keys():
                loss += 0.5 * self.reg * jnp.linalg.norm(self.layers[layer].params['w'])

        self.backward(dout)
        
        if self.reg > 0:
            for layer in self.layers.keys():
                self.layers[layer].grads['w'] += self.reg * self.layers[layer].params['w']
                
        return out, loss
    
    def backward(self,dout):
        for layer in reversed(self.layers.keys()):
            dout = self.layers[layer].backward(dout)
        return dout

    def adamStep(self):
        for layer in self.layers.keys():
            for param, grad in zip(self.layers[layer].params.keys(), self.layers[layer].grads.keys()):
                self.layers[layer].params[param] = self.adam_configs[layer][param].step(self.layers[layer].params[param], self.layers[layer].grads[grad]) 