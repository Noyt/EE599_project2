import argparse
from helpers import generate_train_test, calculate_gain, create_folders_structure
from torch import set_grad_enabled, manual_seed
from train import train
from modules import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='EE-559 Project 2')

parser.add_argument('--run_classification',
                    action='store_true', default=False,
                    help='Runs the experiment as a classification task rather than regression')

parser.add_argument('--seed',
                    type=int, default=42,
                    help='Random seed (default 42, < 0 is no seeding)')

parser.add_argument('--graph',
                    action='store_true', default=False,
                    help='Make a graph of the experiment')

args = parser.parse_args()
manual_seed(args.seed)

# Deactivating autograd
set_grad_enabled(False)

train_data, test_data, train_labels, test_labels = generate_train_test()

if args.run_classification:
    gain = calculate_gain('tanh')

    model = Sequential(
        Linear(2, 25, gain),
        Tanh(),
        Linear(25, 25, gain),
        Tanh(),
        Linear(25, 25, gain),
        Tanh(),
        Linear(25, 25, gain),
        Tanh(),
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

train_losses, val_losses, accuracies = train(model=model,
                                             optimizer=optimizer,
                                             criterion=criterion,
                                             train_set=train_data,
                                             train_target=train_labels,
                                             test_set=test_data,
                                             test_target=test_labels,
                                             epochs=50)


# Graph creation
if args.graph:
    create_folders_structure()

    fig, ax1 = plt.subplots()

    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.tick_params(axis='y')
    ax1.plot(range(1, len(train_losses) + 1), train_losses,
             label='Train Loss')
    ax1.plot(range(1, len(val_losses) + 1), val_losses,
             label='Test Loss')
    ax1.legend(loc='upper left')

    if args.run_classification:
        ax2 = ax1.twinx()

        color = 'tab:red'
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(range(1, len(accuracies) + 1), accuracies,
                 label='Accuracy', color=color)

        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='lower left')
    fig.tight_layout()

    # Saving graph
    name = 'classification' if args.run_classification else 'regression'
    plt.savefig(f'graph/{name}')





