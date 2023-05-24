from torch import nn

from src.model.coder import Coder


class Encoder(Coder):
    def __init__(self, embedding_size, num_tokens, batch_size, num_heads):
        super().__init__(embedding_size, num_tokens, batch_size, num_heads)

        self.layer_norm_1 = nn.LayerNorm([num_tokens, embedding_size])

        self.feed_forward_layer = nn.Linear(512, 512)
        self.activation_function = nn.ReLU()
        self.layer_norm_2 = nn.LayerNorm([num_tokens, embedding_size])

    def forward(self, x):
        z = self.self_attention(x)
        z1 = self.layer_norm_1(x + z)
        z2 = self.activation_function(self.feed_forward_layer(z1))
        return self.layer_norm_2(z1 + z2)
