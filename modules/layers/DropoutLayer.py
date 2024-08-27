from modules.kernels.JAXKernels import kernelDropoutForward, kernelDropoutBackward

class DropoutLayer:
    def __init__(self, p_keep = 1):
        self.cache = None
        self.p_keep = p_keep

    def __call__(self, input_data, mode = "train"):
        return self.forward(input_data, mode)
    
    def forward(self, input_data, mode = "train"):
        """
            Performs the forward pass for (inverted) dropout.

            Inputs:
            - input_data: Input data, of any shape
            - dropout_param: A dictionary with the following keys:
            - p_keep : Dropout parameter. We keep each neuron output with probability p.
            - mode: 'test' or 'train'. If the mode is train, then perform dropout;
                                       If the mode is test, then just return the input.

            Outputs:
            - out: Array of the same shape as input_data.
            - cache: mask - is the dropout mask that was used
              to multiply the input; in test mode, mask is None.
        """
        if mode == "train":
            out, self.cache = kernelDropoutForward(input_data, self.p_keep)
        else:
            out = input_data
        return out
    
    def backward(self, d_out):
        """
            Perform the backward pass for (inverted) dropout.

            Inputs:
            - d_out: Upstream derivatives, of any shape
            - cache: mask from dropout_forward.
        """
        dropout_mask = self.cache
        d_input_data = kernelDropoutBackward(d_out, dropout_mask)
        return d_input_data