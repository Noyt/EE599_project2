from torch import empty, mean, log, sum, exp
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
        # Inherent attributes
        self.in_features = in_features
        self.out_features = out_features
        self.params = empty((in_features, out_features)) # TODO handle proper initialization
        self.bias = empty(out_features)

        # Backward pass information
        self.input = empty(in_features)
        self.output = empty(out_features)

    def forward(self, input: tensor) -> tensor:
        # Forward computation
        output = input.mv(self.params) + self.bias

        # Recording information for backward pass
        self.input = input
        self.output = output

        return output

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return [self.params, self.bias]

class ReLU(Module):
    """
    TODO
    """
    def __init__(self):
        self.input = None

    def forward(self, input: tensor()) -> tensor:
        # Forward computation
        zeros = empty.new_zeros(len(input))
        output = input.where(input > 0, zeros)

        # Recording information for backward pass
        self.input = input
        return output

    def backward(self, *gradwrtoutput):
        ones = empty.new_ones(len(input))
        zeros = empty.new_zeros(len(input))
        derivative = ones.where(self.input > 0, zeros)

        return derivative

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

class MSELoss(Module):
    """
    Mean squared Loss: performs 1/N * sum(prediction - target)
    """
    def __init__(self):
        self.error = None

    def forward(self, predictions, target)->tensor:
        self.error = (predictions-target)
        return mean(self.error.pow(2)) # 1/N * sum((x-y)^2)

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