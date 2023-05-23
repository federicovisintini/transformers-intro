from torch import nn
from torch.nn import functional as F


class Embedding(nn.Module):
    def __init__(self, vocabulary_size, embedding_size=512):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.layer = nn.Linear(vocabulary_size, embedding_size)

    def forward(self, batch):
        x = batch['input_ids']
        one_hot_encoded_x = F.one_hot(x, num_classes=self.vocabulary_size).float()
        return self.layer(one_hot_encoded_x)


class Transformer(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

        self.layer1 = nn.Linear(512, 32)
        self.fn1 = nn.ReLU()
        self.layer2 = nn.Linear(32, 4)
        self.fn2 = nn.Softmax()

    def forward(self, x):
        token = x
        embedded_x = self.embedding(token)

        out1 = self.fn1(self.layer1(embedded_x))
        return self.fn2(self.layer2(out1))
