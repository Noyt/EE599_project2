from torch import empty
from torch import tensor
import math


def generate_train_test(nb_pairs=1000):
    """
    Generates nb_pairs 2-dimensional data points with coordinates in [0,1]
    and corresponding label of appartenance to circle centered on [0.5, 0.5] of radius 1/sqrt(2pi),
    for both training and testing sets
    """
    labels = empty(nb_pairs * 2, 1)
    inputs = empty((nb_pairs * 2, 2, 1)).uniform_()

    max_radius = 1 / ((2 * math.pi) ** (1 / 2))
    for i, input_ in enumerate(inputs):
        x, y = input_[0] - 0.5, input_[1] - 0.5
        labels[i] = (x ** 2 + y ** 2) ** (1 / 2) <= max_radius

    train_data, train_labels = inputs[:nb_pairs], labels[:nb_pairs]
    test_data, test_labels = inputs[nb_pairs:], labels[nb_pairs:]

    return train_data, test_data, train_labels, test_labels


def calculate_gain(activation_name: str):
    """
    Gain to properly adjust weight initialization according to following activation function
    """
    if activation_name.lower() == "relu":
        return math.sqrt(2)
    elif activation_name.lower() == 'tanh':
        return  5.0/3.0
    else:
        return 1


def softmax(input: tensor):
    """Computes the softmax of the input which is : softmax(i) = exp(input(i))/sum(exp(input(j)))"""
    return input.exp()/sum(input.exp())