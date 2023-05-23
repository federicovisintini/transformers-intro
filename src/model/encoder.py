import math

import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, embedding_size, num_tokens, batch_size, num_heads):
        super().__init__()
        assert embedding_size % num_heads == 0
        self.attention_matrix = nn.Parameter(
            torch.randn(num_heads, embedding_size, embedding_size // num_heads), requires_grad=True)

        self.feature_reduction_matrix = nn.Parameter(
            torch.randn(batch_size, embedding_size, 512), requires_grad=True)

        self.layer_norm_1 = nn.LayerNorm([num_tokens, embedding_size])
        self.layer_norm_2 = nn.LayerNorm([num_tokens, embedding_size])

        self.feed_forward_layer = nn.Linear(512, 512)
        self.activation_function = nn.ReLU()

        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_feature = self.embedding_size // self.num_heads

    def self_attention(self, x):
        # following convention:
        # batch_size = 3
        # num_heads = 8
        # num_tokens = 32
        # embedding_size = 512
        # head_feature = embedding_size / num_heads = 64

        q = torch.tensordot(x, self.attention_matrix, dims=([2], [1]))  # 3, 32, 8, 64
        q = torch.swapaxes(q, 1, 2)  # 3, 8, 32, 64
        q = torch.reshape(q, (self.batch_size * self.num_heads, self.num_tokens, self.head_feature))  # 24, 32, 64
        k = v = q

        kt = torch.transpose(k, 1, 2)  # 24, 64, 32
        z = torch.bmm(q, kt)
        z = F.softmax(torch.div(z, math.sqrt(self.head_feature)), dim=2)  # 24, 32, 32
        z = torch.bmm(z, v)  # 24, 32, 64
        z = torch.reshape(z, (self.batch_size, self.num_heads, self.num_tokens, self.head_feature))  # 3, 8, 32, 64
        z = torch.swapaxes(z, 1, 2)  # 3, 32, 8, 64
        z = torch.reshape(z, (self.batch_size, self.num_tokens, self.embedding_size))  # 3, 32, 512

        return torch.bmm(z, self.feature_reduction_matrix)  # 3, 32, 512

    def feed_forward(self, x):
        raise NotImplemented

    def forward(self, x):
        z = self.self_attention(x)

        # ADD and NORMALIZE 1
        z1 = self.layer_norm_1(x + z)

        # FEED FORWARD
        z2 = self.activation_function(self.feed_forward_layer(z1))

        # ADD and NORMALIZE 2
        return self.layer_norm_2(z1 + z2)
