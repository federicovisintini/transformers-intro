import typing as t

from torch import nn

from src.model.embedding import Embedding
from src.model.encoder import Encoder
from src.model.positional_encoder import PositionalEncoder


class Transformer(nn.Module):
    def __init__(
            self,
            embedding: Embedding,
            positional_encoder: PositionalEncoder,
            encoders: t.List[Encoder]
    ):
        super().__init__()
        self.embedding = embedding
        self.positional_encoder = positional_encoder
        self.encoders = nn.ModuleList(encoders)

    def forward(self, token):
        embedded_tokens = self.embedding(token)

        x = self.positional_encoder(embedded_tokens)

        for i, encoder in enumerate(self.encoders):
            print(f"Passing in encoder {i}")
            x = encoder(x)

        return x
