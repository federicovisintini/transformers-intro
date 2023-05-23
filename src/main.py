from torch import nn

from src.dataloader import train_dataloader
from src.model import Transformer, DEVICE
import torch.optim as optim

if __name__ == '__main__':
    model = Transformer()
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, batch in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = batch['translation']['de']
            labels = batch['translation']['en']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
