import modules
from helpers import generate_train_test, calculate_gain
from torch import set_grad_enabled
from train import train
from modules import *

# Deactivating autograd
set_grad_enabled(False)

train_data, test_data, train_labels, test_labels = generate_train_test()

gain = calculate_gain('relu')

model = Sequential(
    Linear(2, 25, gain),
    ReLU(),
    Linear(25, 25, gain),
    ReLU(),
    Linear(25, 25, gain),
    ReLU(),
    Linear(25, 25, gain),
    ReLU(),
    Linear(25, 2)
)

criterion = CrossEntropyLoss()
optimizer = SGD(parameters=model.param(), lr=0.01)


train(model=model,
      optimizer=optimizer,
      criterion=criterion,
      train_set=train_data,
      train_target=train_labels,
      test_set=test_data,
      test_target=test_labels,
      epochs=50)

