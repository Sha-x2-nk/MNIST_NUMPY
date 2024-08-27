from modules.kernels.JAXKernels import kernelReLUForward, kernelReLUBackward

class ReLULayer:
    def __init__(self):
        self.cache = None
    
    def __call__(self, input_data, mode = "train"):
        return self.forward(input_data, mode)
    
    def forward(self, input_data, mode = "train"):
        """
            Computes the forward pass for a layer of rectified linear units (ReLUs).

            Input:
            - input_data: Inputs, of any shape

            Returns and stores:
            - out: Output, of the same shape as x
            - cache: x
        """
        out = kernelReLUForward(input_data)

        if mode == "train":
            self.cache = input_data
        
        return out
    
    def backward(self, d_out):
        """
            Computes the backward pass for a layer of rectified linear units (ReLUs).

            Input:
            - d_out: Upstream derivatives, of any shape
            - cache: Input data, of same shape as dout

            Returns:
            - d_input_data: Gradient with respect to input_data
        """
        input_data = self.cache

        d_input_data = kernelReLUBackward(d_out, input_data)
        return d_input_data