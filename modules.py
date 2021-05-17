from torch import empty
from torch import tensor
from torch.nn import Linear
import math


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
        # Inherent attributes
        self.in_features = in_features
        self.out_features = out_features

        self.weights = empty((out_features, in_features)) # TODO handle proper initialization
        self.bias = empty((out_features, 1))

        # Backward pass information
        self.input = empty((in_features, 1))
        self.output = empty((out_features, 1))

        # Module update information
        self.gradwrtweights = empty((out_features, in_features))
        self.gradwrtbias = empty((out_features, 1))

    def forward(self, input: tensor) -> tensor:
        # Forward computation
        output = self.weights.mv(input) + self.bias

        # Recording information for backward pass
        self.input = input
        self.output = output

        return output

    def backward(self, gradwrtoutput, lr):
        # Computing updates
        self.gradwrtweights = gradwrtoutput @ self.input.transpose(0,-1)
        self.gradwrtbias = gradwrtoutput

        # Propagation
        gradwrtinput = self.weights.transpose(-1,0) @ gradwrtoutput

        return gradwrtinput

    def update(self, func, **kwargs):
        self.weights = func(self.weights, self.gradwrtweights, kwargs)
        self.bias = func(self.bias, self.gradwrtbias, kwargs)

    def param(self):
        return [self.weights, self.bias]


class ReLU(Module):
    """
    TODO
    """
    def __init__(self):
        self.input = None

    def forward(self, input: tensor) -> tensor:
        # Forward computation
        zeros = empty.new_zeros(len(input))
        output = input.where(input > 0, zeros)

        # Recording information for backward pass
        self.input = input
        return output

    def backward(self, gradwrtinput):
        ones = empty.new_ones(len(input))
        zeros = empty.new_zeros(len(input))
        derivative = ones.where(self.input > 0, zeros)

        gradwrtoutput = gradwrtinput*derivative

        return  gradwrtoutput

    def param(self):
        return []


class Tanh(Module):
    def __init__(self):
        self.input = None

    def forward(self, *input):
        # Forward computation
        output = input.apply_(math.tanh)

        # Recording information for backward pass
        self.input = input
        return output

    def backward(self, gradwrtinput):
        derivative = input.apply_(lambda x: 1.0/(math.cosh(x)**2))
        gradwrtoutput = gradwrtinput * derivative
        return gradwrtoutput

    def param(self):
        return []


class Sequential(Module):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []