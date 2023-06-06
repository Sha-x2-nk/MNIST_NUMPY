import numpy as np
import math

class affineLayer:
    def __init__(self, in_features, out_features):
        self.params = { 'w' : np.random.randn(in_features,out_features)*np.sqrt(2/in_features),  #custom initialisation
                        'b' : np.zeros(out_features)
                      }
        self.grads = {'w':0,'b':0}
        self.cache = None

    def __call__(self,x):
        return self.forward(x)

    def forward(self,x,mode = "train"):
        """Computes the forward pass for an affine (fully connected) layer.

        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.

        Inputs:
        - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
        - w: A numpy array of weights, of shape (D, M)
        - b: A numpy array of biases, of shape (M,)

        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        _x = x.reshape(x.shape[0], -1)
        out = _x@self.params["w"] + self.params["b"]
        self.cache = (x,self.params["w"], self.params["b"])
        return out

    def backward(self,dout):
        """Computes the backward pass for an affine (fully connected) layer.

        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
        - x: Input data, of shape (N, d_1, ... d_k)
        - w: Weights, of shape (D, M)
        - b: Biases, of shape (M,)

        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x,w,b = self.cache
        db = dout.sum(axis = 0)
        _x = x.reshape(x.shape[0], -1)
        dw = _x.T@dout
        dx = dout@(w.T)
        dx = dx.reshape(*x.shape)
        self.grads['w'] = dw
        self.grads['b'] = db
        return dx

class ReLULayer:
    def __init__(self):
        self.cache = None

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self,x):
        """Computes the forward pass for a layer of rectified linear units (ReLUs).

        Input:
        - x: Inputs, of any shape

        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
        """
        out = np.maximum(0,x)
        self.cache = x
        return out

    def backward(self,dout):
        """Computes the backward pass for a layer of rectified linear units (ReLUs).

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
        """
        x = self.cache
        dx = dout*(x>0)
        return dx

class dropoutLayer:
    def __init__(self, p_keep=1):
        self.cache = None
        self.p_keep = p_keep

    def __call__(self ,x, mode="train"):
        return self.forward(x,mode)
    
    def forward(self, x, mode="train"):
        """
        Performs the forward pass for (inverted) dropout.

        Inputs:
        - x: Input data, of any shape
        - dropout_param: A dictionary with the following keys:
        - p_keep : Dropout parameter. We keep each neuron output with probability p.
        - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.

        Outputs:
        - out: Array of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
        mask that was used to multiply the input; in test mode, mask is None.
        """
        dropout_mask = None
        if mode == "train":
            dropout_mask = (np.random.rand(*x.shape)<self.p_keep)/self.p_keep
            out = dropout_mask*x
            self.cache = dropout_mask
        else:
            out = x
        return out
    
    def backward(self, dout):
        """
        Perform the backward pass for (inverted) dropout.

        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from dropout_forward.
        """
        dropout_mask = self.cache
        dx = dout*dropout_mask
        return dx

class affineReLULayer():

    def __init__(self, in_features, out_features, p_keep = 1): #p_keep is probablity of keeping a neural in layer(for dropout)
        self.fc = affineLayer(in_features,out_features)
        self.relu = ReLULayer()
        self.dropout = None
        if p_keep>=0 and p_keep<1:
            self.dropout = dropoutLayer(p_keep)
        self.params = self.fc.params
        self.grads = self.fc.grads

    def __call__(self, x):
        return self.forward(x)
    
    def get_grads(self):
        return self.fc.get_grads()

    def forward(self,x,mode = "train"):
        out = self.fc(x)
        if self.dropout:
            out = self.dropout(out,mode=mode)
        out = self.relu(out)
        return out
    
    def backward(self, dout):
        dout = self.relu.backward(dout) 
        if self.dropout:
            dout = self.dropout.backward(dout)
        dout = self.fc.backward(dout)
        return dout
    
class NeuralNetwork:
    def __init__(self,reg = 0):
        self.layers = {
            'l1': affineReLULayer(784, 2048,p_keep = 0.7),
            'l2': affineLayer(2048,10)
        }
        self.mode = "test"
        self.adam_configs = {}
         
        for layer in self.layers.keys():
            self.adam_configs[layer] = {}
            for param in self.layers[layer].params:
                self.adam_configs[layer][param] = {}
        
        self.reg = reg
    def load_saved(self):
        for layer in self.layers.keys():
            for param in self.layers[layer].params.keys():
                self.layers[layer].params[param] = np.load('saved_/'+str(layer)+'_'+str(param)+'.npy')
    def save_model(self):
        for layer in self.layers.keys():
            for param in self.layers[layer].params.keys():
                np.save('saved_/'+str(layer)+'_'+str(param)+'.npy',self.layers[layer].params[param])
    def train(self):
        self.mode = "train"
    def eval(self):
        self.mode = "eval"

    def __call__(self, x, y =None):
        return self.forward(x,y)

    def forward(self,x, y= None):
        out = x
        for layer in self.layers.keys():
            out = self.layers[layer].forward(out,mode = self.mode)
        if self.mode == "eval" or y is None:
            return out
        loss, dout = self.softmaxLoss(out,y)
        
        if self.reg > 0:
            for layer in self.layers.keys():
                loss += 0.5*self.reg*(np.sum(self.layers[layer].params["w"]*self.layers[layer].params["w"]))

        self.backward(dout)
        
        if self.reg > 0:
            for layer in self.layers.keys():
                self.layers[layer].grads['w'] += self.reg*self.layers[layer].params['w']
        return out, loss
    
    def backward(self,dout):
        for layer in reversed(self.layers.keys()):
            dout = self.layers[layer].backward(dout)
        return dout

    def softmaxLoss(self, x, y):
        """Computes the loss and gradient for softmax classification.

        Inputs:
        - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
        class for the ith input.
        - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
        0 <= y[i] < C

        Returns a tuple of:
        - loss: Scalar giving the loss
        - dx: Gradient of the loss with respect to x
        """
        shifted_x = x - np.max(x,axis = 1, keepdims=True)
        shifted_x = shifted_x 
        exp_x = np.exp(shifted_x)

        denom = np.sum(exp_x,axis = 1, keepdims=True) 
        
        scores = exp_x/denom
        scores = scores + 1e-10 #epsilon to prevent -log(0)
        N = x.shape[0]
        
        loss = np.sum(-np.log(scores[np.arange(N), y]))/N

        dx = scores
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx

    def adamStep(self):
        for layer in self.layers.keys():
            for param in self.layers[layer].params.keys():
                adam_config = self.adam_configs[layer][param]

                adam_config.setdefault('learning_rate', 1e-3)
                adam_config.setdefault('beta1', 0.9)
                adam_config.setdefault('beta2', 0.999)
                adam_config.setdefault('epsilon', 1e-8)
                adam_config.setdefault('m', np.zeros_like(self.layers[layer].params[param]))
                adam_config.setdefault('v', np.zeros_like(self.layers[layer].params[param]))
                adam_config.setdefault('t', 0)
                eps, learning_rate = adam_config['epsilon'],adam_config['learning_rate']
                beta1, beta2 = adam_config['beta1'], adam_config['beta2']
                m,v,t = adam_config['m'],adam_config['v'],adam_config['t']

                t = t+1
                m = beta1*m + (1-beta1)*self.layers[layer].grads[param]
                mt = m/(1-beta1**t) #bias correction
                
                v = beta2*v + (1-beta2)*(self.layers[layer].grads[param]*self.layers[layer].grads[param])
                vt = v/(1-beta2**t) #bias correction

                self.layers[layer].params[param] = self.layers[layer].params[param] - learning_rate*mt/(np.sqrt(vt)+eps)
                #update values
                adam_config['m'],adam_config['v'],adam_config['t'] = m,v,t
                self.adam_configs[layer][param] = adam_config   