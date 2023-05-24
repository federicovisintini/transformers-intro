import typing as t

import torch
from torch import nn

from src.model.decoder import Decoder
from src.model.embedding import Embedding
from src.model.encoder import Encoder
from src.model.positional_encoder import PositionalEncoder


class Transformer(nn.Module):
    def __init__(
            self,
            embedding: Embedding,
            positional_encoder: PositionalEncoder,
            encoders: t.List[Encoder],
            decoders: t.List[Decoder],
            embedding_size: int,
            vocabulary_size: int,
    ):
        super().__init__()
        self.embedding = embedding
        self.positional_encoder = positional_encoder
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)

        self.qkv_matrix_dim = self.encoders[0].qkv_matrix_dim
        self.k_matrix = nn.Parameter(torch.randn(*self.qkv_matrix_dim), requires_grad=True)
        self.v_matrix = nn.Parameter(torch.randn(*self.qkv_matrix_dim), requires_grad=True)

        self.feed_forward_layer = nn.Linear(embedding_size, vocabulary_size)
        self.softmax = nn.Softmax()

    def forward(self, token):
        embedded_tokens = self.embedding(token)

        x = self.positional_encoder(embedded_tokens)

        for i, encoder in enumerate(self.encoders):
            x = encoder(x)  # 3, 32, 512

        x = torch.reshape(x, (3, 32, 8, 64))
        x = torch.swapaxes(x, 1, 2)
        x = torch.reshape(x, (24, 32, 64))

        k = self.encoders[0].get_qkv(x, self.k_matrix)
        v = self.encoders[0].get_qkv(x, self.v_matrix)

        for i, decoder in enumerate(self.decoders):
            x = decoder(x, k, v)

        return self.softmax(self.feed_forward_layer(x))
