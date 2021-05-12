from torch import empty
from torch import tensor


class Module(object):
    """
    Abstract Module class
    """
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Linear(Module):
    """
    Fully connected layer, performs linear combinations of input and adds bias, for the required number of outputs
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        # TODO handle proper initialization
        self.params = empty((in_features, out_features))
        self.bias = empty(out_features)

    def forward(self, input: tensor) -> tensor:
        s = input.mv(self.params) + self.bias
        return s

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return [self.params, self.bias]


class ReLU(Module):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Tanh(Module):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Sequential(Module):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []