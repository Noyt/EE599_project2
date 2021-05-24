from torch import empty
from torch import tensor

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

    def __init__(self, in_features, out_features, gain=1.0):
        # Inherent attributes
        self.in_features = in_features
        self.out_features = out_features

        # Weight initialization (Xavier)
        self.weights = empty((out_features, in_features))
        std = gain * math.sqrt(2.0 / (in_features + out_features))
        self.weights.normal_(0, std)
        self.bias = empty((out_features, 1)).new_zeros((out_features, 1))

        # Backward pass information
        self.input = empty((in_features, 1))
        self.output = empty((out_features, 1))

        # Module update information
        self.gradwrtweights = empty((out_features, in_features))
        self.gradwrtbias = empty((out_features, 1))

    def forward(self, input: tensor) -> tensor:
        # Forward computation
        output = self.weights @ (input) + self.bias

        # Recording information for backward pass
        self.input = input
        self.output = output

        return output

    def backward(self, gradwrtoutput):
        # Computing updates
        self.gradwrtweights = gradwrtoutput @ self.input.transpose(0, -1)
        self.gradwrtbias = gradwrtoutput

        # Propagation
        gradwrtinput = self.weights.transpose(-1, 0) @ gradwrtoutput

        return gradwrtinput

    def update(self, func, kwargs):
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
        zeros = empty(input.size()).new_zeros(input.size())
        output = input.where(input > 0, zeros)

        # Recording information for backward pass
        self.input = input
        return output

    def backward(self, gradwrtinput):
        ones = empty(self.input.size()).new_ones(self.input.size())
        zeros = empty(self.input.size()).new_zeros(self.input.size())
        derivative = ones.where(self.input > 0, zeros)

        gradwrtoutput = gradwrtinput * derivative

        return gradwrtoutput

    def param(self):
        return []


class Tanh(Module):
    def __init__(self):
        self.input = None

    def forward(self, input):
        # Forward computation
        output = input.clone().apply_(math.tanh)

        # Recording information for backward pass
        self.input = input
        return output

    def backward(self, gradwrtinput):
        derivative = self.input.clone().apply_(lambda x: 1.0 / (math.cosh(x) ** 2))
        gradwrtoutput = gradwrtinput * derivative
        return gradwrtoutput

    def param(self):
        return []


class Sequential(Module):

    def __init__(self, *modules: Module):  # Need to pass Module in Sequential
        self.seq = list()
        for module in modules:
            self.seq.append(module)

    def forward(self, input) -> tensor:
        for module in self.seq:
            output = module.forward(input)
            input = output
        return output

    def backward(self, gradwrtoutput):
        output = gradwrtoutput
        for module in self.seq[::-1]:  # Goes through module in reverse order
            output = module.backward(output)

    def param(self):
        linear_layers = []
        for module in self.seq:
            if isinstance(module, Linear):
                linear_layers.append(module)
        return linear_layers


class SGD(Module):
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for module in self.parameters:
            #module.update(lambda args, gradients, lr: args - lr * gradients,
            module.update(lambda args, gradients, kwargs: args - kwargs['lr'] * gradients,
                          {"lr": self.lr})  # Pass the SGD func to the layers


class MSELoss(Module):
    """
    Mean squared Loss: performs (prediction - target)^2
    """

    def __init__(self):
        self.error = None

    def forward(self, predictions, target) -> tensor:
        self.error = (predictions - target)
        return self.error.pow(2)  # (x-y)^2

    def backward(self):
        return 2 * self.error  # grad error = 2 * (x-y)

    def param(self):
        return []


class CrossEntropyLoss(Module):
    """
    Cross Entropy Loss: Criterion that combines LogSoftmax and NLLLoss in one single class.
    """

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, predictions, actual) -> tensor:
        loss = empty(1).zero_()
        self.x = predictions
        self.y = actual
        for i in range(len(actual)):
            loss += -predictions[int(actual[i].item())] + (predictions.exp()).sum().log()

        return loss / len(actual)

    def backward(self):
        grad = empty(self.x.shape)
        for i in range(len(self.x)):
            grad[i] = self.x[i].exp() / self.x.exp().sum()  # The derivation is this for each value of x_i

        grad[int(self.y.item())] -= 1  # Need to subtract by -1 for the sample that is correct
        return grad / len(self.y)  # Need to normalize
