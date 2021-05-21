from modules import CrossEntropyLoss
import torch
import torch.nn.functional as F

def train(model, optimizer, criterion, train_set, train_target, test_set, test_target, epochs=100):
    train_losses, val_losses, accuracies = [],[],[]

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        # Training
        for input, target in zip(train_set, train_target):
            model_output = model.forward(input)
            train_loss += criterion.forward(model_output, target).data.item()
            model.backward(criterion.backward())
            optimizer.step()

        # Validation
        num_correct = 0
        num_examples = 0
        for input, target in zip(test_set, test_target):
            model_output = model.forward(input)
            val_loss += criterion.forward(model_output, target).data.item()
            correct = torch.eq(torch.max(F.softmax(model_output, dim= 0), dim=1)[1], target).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]



        print(
            f'Epoch {epoch}, Training Loss: {train_loss:.2f}, Validation Loss : {val_loss:.2f}')#, accuracy = {num_correct / num_examples:.2f}')

        print(f"Accuracy : {num_correct/num_examples}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        #accuracies.append(num_correct / num_examples)

    return train_losses, val_losses, accuracies