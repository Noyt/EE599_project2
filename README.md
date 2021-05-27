# EE559 Mini Project 2
Mini deep-learning framework

This repository contains a mini “deep learning framework” using only pytorch’s
tensor operations and the standard math library, hence in particular without using autograd or the
neural-network modules.

A ``test.py`` file can be run from a terminal with the command ``python test.py``

This will do the following:

* Generate a training and a test set of 1, 000 points sampled uniformly in [0, 1]^2 , each with a
label 0 if outside the disk centered at (0.5, 0.5) of radius 1/√2π, and 1 inside
* Build a network with two input units, one output unit, three hidden layers of 25 units, connected by ReLU activations
* Train it with MSE, logging the loss,
* Compute and print the final train and the test errors

Alternatively, this task can be run as a classification problem, by running the command ``python test.py --run_classification``

This will do the following:

* Generate a training and a test set of 1, 000 points sampled uniformly in [0, 1]^2 , each with a
label 0 if outside the disk centered at (0.5, 0.5) of radius 1/√2π, and 1 inside
* Build a network with two input units, two output unit, three hidden layers of 25 units, connected by Tanh activations
* Train it with MSE, logging the loss,
* Compute and print the final train and the test errors, and test accuracy

Adding a ``--graph`` option to either of these commands will generate a graph with the corresponding metrics. 