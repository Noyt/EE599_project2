import torch
from modules import *


class TestLinear:
    def test_lin_backward(self):
        lin = Linear(2, 3)
        lin.weights = torch.ones((3, 2))
        lin.bias = torch.ones((3, 1))
        lin.input = torch.ones((2, 1))

        gradwrtoutput = torch.ones((3, 1)) * 2
        gradwrtinput = lin.backward(gradwrtoutput)
        assert torch.equal(gradwrtinput, torch.ones((2, 1)) * 6)

    def test_sequential_mse_backprop(self):
        lin1 = Linear(2, 3)
        lin1.weights = torch.ones((3, 2))
        lin1.bias = torch.ones((3, 1))

        lin2 = Linear(3, 1)
        lin2.weights = torch.ones((1, 3))
        lin2.bias = torch.ones((1, 1))

        input = torch.ones((2, 1))
        target = torch.tensor(([[9.0]]))
        model = Sequential(lin1, ReLU(), lin2)
        mse = MSELoss()
        optimizer = SGD(parameters=model.param(), lr=0.01)

        model_output = model.forward(input)
        mse.forward(model_output, target)
        model.backward(mse.backward())

        optimizer.step()

        assert model_output == 10
        assert lin1.weights.equal(torch.tensor([[0.98, 0.98],
                                                [0.98, 0.98],
                                                [0.98, 0.98]]))
        assert lin1.bias.equal(torch.tensor([[0.98],
                                             [0.98],
                                             [0.98]]))
        assert lin2.weights.equal(torch.tensor([[0.94, 0.94, 0.94]]))
        assert lin2.bias.equal(torch.tensor([[0.98]]))

    def test_dummy_loss(self):
        mse = MSELoss()
        loss = mse.forward(torch.tensor([[1.0]]), torch.tensor([[0.0]]))
        assert 1 == loss
        assert 2 == mse.backward()

    def test_activations(self):
        input = torch.tensor([[2.0, 2.0],
                              [2.0, 2.0],
                              [-1.0, -1.0]])
        ones = torch.ones((3,2))
        relu = ReLU()
        output = relu.forward(input)
        assert output.equal(torch.tensor([[2.0, 2.0],
                                            [2.0, 2.0],
                                            [0.0, 0.0]]))
        backres = relu.backward(ones)
        assert backres.equal(torch.tensor([[1.0, 1.0],
                                            [1.0, 1.0],
                                            [0.0, 0.0]]))
        tanh = Tanh()
        output = tanh.forward(input)
        tanh2 = math.tanh(2.0)
        tanh1 = math.tanh(-1.0)
        assert output.equal(torch.tensor([[tanh2, tanh2],
                                          [tanh2, tanh2],
                                          [tanh1, tanh1]]))

        backres = tanh.backward(ones)
        dev_tanh2 = 1 / (math.cosh(2.0) ** 2)
        dev_tanh1 = 1 / (math.cosh(-1.0) ** 2)
        assert backres.equal(torch.tensor([[dev_tanh2, dev_tanh2],
                                           [dev_tanh2, dev_tanh2],
                                           [dev_tanh1, dev_tanh1]]))

class TestModel:
    def test_model(self):
        model = Sequential(Linear(1, 1), ReLU())
        optimizer = SGD(parameters=model.param(), lr=0.1)
        input = torch.tensor([[1]])
        true = torch.tensor([[0]])

        model_output = model.forward(input)
        print(model_output)
