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
            num_tokens: int,
            vocabulary_size: int,
    ):
        super().__init__()
        self.embedding = embedding
        self.positional_encoder = positional_encoder
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.hidden_dim_final_layer = 512  # this is arbitrary, does not need to be embedding_size

        self.qkv_matrix_dim = self.encoders[0].qkv_matrix_dim
        self.k_matrix = nn.Parameter(torch.randn(*self.qkv_matrix_dim), requires_grad=True)
        self.v_matrix = nn.Parameter(torch.randn(*self.qkv_matrix_dim), requires_grad=True)

        self.feed_forward_layer1 = nn.Linear(embedding_size * num_tokens, self.hidden_dim_final_layer)
        self.activation_function = nn.ReLU()
        self.feed_forward_layer2 = nn.Linear(self.hidden_dim_final_layer, vocabulary_size)
        self.softmax = nn.Softmax(dim=1)

    def final_layer(self, x):
        x = torch.flatten(x, start_dim=1)

        z = self.feed_forward_layer1(x)
        x1 = self.activation_function(z)

        z1 = self.feed_forward_layer2(x1)
        return self.softmax(z1)

    def forward(self, token):
        embedded_tokens = self.embedding(token)

        x = self.positional_encoder(embedded_tokens)

        for i, encoder in enumerate(self.encoders):
            x = encoder(x)  # 3, 32, 512

        k = self.encoders[0].get_qkv(x, self.k_matrix)
        v = self.encoders[0].get_qkv(x, self.v_matrix)

        # TODO initial decoder does not take x as input
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, k, v)

        return self.final_layer(x)
