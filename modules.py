from torch import empty
from math import log, exp
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

        return gradwrtoutput

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

    def __init__(self, *modules : Module): #Need to pass Module in Sequential
        self.seq = list()
        for module in modules:
            self.seq.append(module)

    def forward(self, input) -> tensor :
        for module in self.seq:
            output = module.forward(input)
            input = output
        return output

    def backward(self, *gradwrtoutput):
        output = gradwrtoutput
        for module in self.seq[::-1]: #Goes through module in reverse order
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
            module.update(lambda args, gradients, lr: args - lr * gradients, {"lr": self.lr}) #Pass the SGD func to the layers


class MSELoss(Module):
    """
    Mean squared Loss: performs 1/N * sum(prediction - target)
    """
    def __init__(self):
        self.error = None

    def forward(self, predictions, target)->tensor:
        self.error = (predictions-target)
        return self.error.pow(2).mean() # 1/N * sum((x-y)^2)

    def backward(self):
        return (2/len(self.error))* self.error #grad error = 2/N * (x-y)

    def param(self):
        return []

class CrossEntropyLoss(Module):
    """
    Cross Entropy Loss: Criterion that combines LogSoftmax and NLLLoss in one single class.
    """
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, predictions, actual)->tensor:
        loss = empty(1).zero_()
        self.x = predictions
        self.y = actual
        for i in range(len(actual)):
            loss += -predictions[i, actual[i]] + log(sum(exp(predictions[i])))
        return loss/len(actual)

    def backward(self):
        grad = empty(self.x.shape)
        for i in range(len(self.x)):
            row = self.x[i]
            pred = self.y[i]
            grad[i] = 1/sum(exp(row)) * exp(row) #The derivation is this for each value of x_i
            grad[i, pred] -= 1 #Need to subtract by -1 for the sample that is correct

        return grad/len(self.y) #Need to normalize