from torch import empty
from torch import tensor
from torch.nn import Linear


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

        self.weights = empty((out_features, in_features)) # TODO handle proper initialization + CHANGE in/out stuff
        self.bias = empty(out_features)

        # Backward pass information
        self.input = empty(in_features)
        self.output = empty(out_features)

    def forward(self, input: tensor) -> tensor:
        # Forward computation
        output = self.weights.mv(input) + self.bias

        # Recording information for backward pass
        self.input = input
        self.output = output

        return output

    def backward(self, gradwrtoutput, lr):
        # Computing updates
        gradwrtweights = gradwrtoutput @ self.input.transpose(0,-1) # TODO check dimensions match
        gradwrtbias = gradwrtoutput

        # Propagation
        gradwrtinput = self.weights.transpose(-1,0) @ gradwrtoutput

        # Updating parameters
        self.weights -= lr * gradwrtweights
        self.bias -= lr * gradwrtbias

        return gradwrtinput

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