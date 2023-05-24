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

        self.qkv_matrix_dim = self.encoders[0].qkv_matrix_dim
        self.k_matrix = nn.Parameter(torch.randn(*self.qkv_matrix_dim), requires_grad=True)
        self.v_matrix = nn.Parameter(torch.randn(*self.qkv_matrix_dim), requires_grad=True)

        self.feed_forward_layer1 = nn.Linear(embedding_size * num_tokens, 512)
        self.activation_function = nn.ReLU()
        self.feed_forward_layer2 = nn.Linear(512, vocabulary_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, token):
        embedded_tokens = self.embedding(token)

        x = self.positional_encoder(embedded_tokens)

        for i, encoder in enumerate(self.encoders):
            x = encoder(x)  # 3, 32, 512

        k = self.encoders[0].get_qkv(x, self.k_matrix)
        v = self.encoders[0].get_qkv(x, self.v_matrix)

        for i, decoder in enumerate(self.decoders):
            x = decoder(x, k, v)

        x = torch.reshape(x, (3, 32 * 512))

        z = self.activation_function(self.feed_forward_layer1(x))
        return self.softmax(self.feed_forward_layer2(z))
