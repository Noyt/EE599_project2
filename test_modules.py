import torch

from modules import *


class TestLinear:
    def test_backward(self):
        lin = Linear(2,3)
        lin.weights = torch.ones((3,2))
        lin.bias = torch.ones((3,1))
        lin.input = torch.ones((2,1))

        gradwrtoutput = torch.ones((3,1))*2
        lr = 0.01
        gradwrtinput = lin.backward(gradwrtoutput, lr)
        assert torch.equal(gradwrtinput, torch.ones((2,1))*6)
        assert torch.equal(lin.weights, torch.ones((3,2))*0.98) # Weight update
        assert torch.equal(lin.bias, torch.ones((3,1))*0.98)# bias update
