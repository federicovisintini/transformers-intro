import torch

from src.dataloader import train_dataloader, VOCABULARY_SIZE
from src.device import DEVICE
from src.model import Embedding

if __name__ == '__main__':
    embedding = Embedding(vocabulary_size=VOCABULARY_SIZE)
    embedding.to(DEVICE)

    for epoch in range(1):

        running_loss = 0.0
        for i, batch in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            x = embedding(batch)
            print(x.size())  # 3, 32, 512

            y = torch.sum(x, dim=2)
            print(y.size())
            print(y)

            break

    print('Finished Training')
