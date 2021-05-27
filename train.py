from modules import CrossEntropyLoss
import helpers


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
        train_loss = train_loss/len(train_set)

        # Validation
        num_correct = 0
        num_examples = len(test_target)
        for input, target in zip(test_set, test_target):
            model_output = model.forward(input)
            val_loss += criterion.forward(model_output, target).data.item()
            if type(criterion) == CrossEntropyLoss:
                found_val = helpers.softmax(model_output).argmax()
                if found_val == target.item():
                    num_correct += 1

        val_loss = val_loss/len(test_set)

        if type(criterion) == CrossEntropyLoss:
            print(
                f'Epoch {epoch}, Training Loss: {train_loss:.2f}, Validation Loss : {val_loss:.2f}, accuracy = {num_correct / num_examples:.2f}')
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            accuracies.append(num_correct / num_examples)
        else:
            print(
                f'Epoch {epoch}, Training Loss: {train_loss:.2f}, Validation Loss : {val_loss:.2f}')
            train_losses.append(train_loss)
            val_losses.append(val_loss)

    return train_losses, val_losses, accuracies