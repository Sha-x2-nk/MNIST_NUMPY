from modules.layers.AffineLayer import AffineLayer
from modules.layers.ReLULayer import ReLULayer
from modules.layers.DropoutLayer import DropoutLayer

class AffineReLULayer:
    # p_keep is probablity of keeping a neural in layer(for dropout)
    def __init__(self, in_features, out_features, p_keep = 1): 
        self.fc = AffineLayer(in_features,out_features)
        self.relu = ReLULayer()

        self.dropout = None
        if p_keep>=0 and p_keep<1:
            self.dropout = DropoutLayer(p_keep)

        self.params = self.fc.params
        self.grads = self.fc.grads

    def __call__(self, input_data, mode = "train"):
        return self.forward(input_data, mode)
    
    def forward(self, input_data, mode = "train"):
        out = self.fc(input_data, mode)

        if self.dropout:
            out = self.dropout(out, mode)

        out = self.relu(out, mode)
        return out
    
    def backward(self, dout):
        dout = self.relu.backward(dout)

        if self.dropout:
            dout = self.dropout.backward(dout)
            
        dout = self.fc.backward(dout)
        return dout