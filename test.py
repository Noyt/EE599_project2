import argparse
from helpers import generate_train_test, calculate_gain
from torch import set_grad_enabled, manual_seed
from train import train
from modules import *


parser = argparse.ArgumentParser(description='EE-559 Project 2')


parser.add_argument('--run_classification',
                    action='store_true', default=False,
                    help = 'Runs the experiment as a classification task rather than regression')

parser.add_argument('--seed',
                    type = int, default = 42,
                    help = 'Random seed (default 42, < 0 is no seeding)')

args = parser.parse_args()
manual_seed(args.seed)


# Deactivating autograd
set_grad_enabled(False)

train_data, test_data, train_labels, test_labels = generate_train_test()

if args.run_classification:
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

else:
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
        Linear(25, 1)
    )

    criterion = MSELoss()

optimizer = SGD(parameters=model.param(), lr=0.01)

train(model=model,
      optimizer=optimizer,
      criterion=criterion,
      train_set=train_data,
      train_target=train_labels,
      test_set=test_data,
      test_target=test_labels,
      epochs=50)

