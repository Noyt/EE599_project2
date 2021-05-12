
import modules
from helpers import generate_train_test
from torch import set_grad_enabled

# Deactivating autograd
set_grad_enabled(False)

train_data, test_data, train_labels, test_labels = generate_train_test()
