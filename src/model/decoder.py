import torch
from torch import nn

from src.model.coder import Coder


class Decoder(Coder):
    def __init__(self, embedding_size, num_tokens, batch_size, num_heads):
        super().__init__(embedding_size, num_tokens, batch_size, num_heads)
        self.q2_matrix = nn.Parameter(torch.randn(*self.qkv_matrix_dim), requires_grad=True)

        self.layer_norm_1 = nn.LayerNorm([num_tokens, embedding_size])
        self.layer_norm_2 = nn.LayerNorm([num_tokens, embedding_size])

        self.feed_forward_layer = nn.Linear(512, 512)
        self.activation_function = nn.ReLU()
        self.layer_norm_3 = nn.LayerNorm([num_tokens, embedding_size])

    def encoder_decoder_attention(self, x, k, v):
        q = self.get_qkv(x, self.q2_matrix)

        z = self.qkv_product(q, k, v)

        return torch.bmm(z, self.feature_reduction_matrix)  # 3, 32, 512

    def forward(self, x, k_from_encoder, v_from_encoder):
        z = self.self_attention(x)
        z1 = self.layer_norm_1(x + z)
        z2 = self.encoder_decoder_attention(z1, k_from_encoder, v_from_encoder)
        z3 = self.layer_norm_2(z1 + z2)
        z4 = self.activation_function(self.feed_forward_layer(z3))
        return self.layer_norm_3(z3 + z4)
