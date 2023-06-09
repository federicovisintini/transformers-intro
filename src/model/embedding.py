from torch import nn
from torch.nn import functional as F


class Embedding(nn.Module):
    def __init__(self, vocabulary_size, embedding_size):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.layer = nn.Linear(vocabulary_size, embedding_size)

    def forward(self, input_token_ids):
        one_hot_encoded_x = F.one_hot(input_token_ids, num_classes=self.vocabulary_size).float()
        return self.layer(one_hot_encoded_x) * self.embedding_size ** 0.5
