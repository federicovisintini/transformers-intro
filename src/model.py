import torch
from torch import nn
from torch.nn import functional as F

from src.device import DEVICE

POSITIONAL_ENCODING_COEFFICIENTS = 300


class Embedding(nn.Module):
    def __init__(self, vocabulary_size, embedding_size):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.layer = nn.Linear(vocabulary_size, embedding_size)

    def forward(self, batch):
        x = batch['input_ids']
        one_hot_encoded_x = F.one_hot(x, num_classes=self.vocabulary_size).float()
        return self.layer(one_hot_encoded_x)


class PositionalEncoder(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, num_tokens, positional_encoding_scalar, batch_size):
        super().__init__()

        assert embedding_size % 2 == 0

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.positional_encoding_scalar = positional_encoding_scalar
        self.batch_size = batch_size

        self.device = None
        self.positional_encoding_vector = self.compute_positional_encoding_vector()

    def compute_positional_encoding_vector(self):
        v1 = torch.arange(self.num_tokens)
        v2 = torch.pow(self.positional_encoding_scalar,
                       - 2 * torch.arange(self.embedding_size // 2) / self.embedding_size)

        prod = torch.outer(v1, v2)

        sin = torch.sin(prod)
        cos = torch.cos(prod)

        data = torch.stack([sin, cos], dim=2).view(self.num_tokens, self.embedding_size)

        return data.repeat(self.batch_size, 1, 1)

    def forward(self, embedded_token):
        if self.device:
            return embedded_token + self.positional_encoding_vector.to(DEVICE) / POSITIONAL_ENCODING_COEFFICIENTS

        return embedded_token + self.positional_encoding_vector / POSITIONAL_ENCODING_COEFFICIENTS

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = args[0]


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
