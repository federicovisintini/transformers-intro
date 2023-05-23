from torch import nn


class Transformer(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

        self.layer1 = nn.Linear(self.embedding.embedding_size, 32)
        self.fn1 = nn.ReLU()
        self.layer2 = nn.Linear(32, 4)
        self.fn2 = nn.Softmax()

    def forward(self, token):
        embedded_x = self.embedding(token)

        out1 = self.fn1(self.layer1(embedded_x))
        return self.fn2(self.layer2(out1))
