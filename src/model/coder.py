import torch
from torch import nn
from torch.nn import functional as F


class Coder(nn.Module):
    def __init__(self, embedding_size, num_tokens, batch_size, num_heads):
        super().__init__()
        assert embedding_size % num_heads == 0,\
            "EMBEDDING_SIZE must be a multiple of NUM_HEADS (multi-head attention)"

        # parameters
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_feature = self.embedding_size // self.num_heads
        self.dim_model = embedding_size
        self.qkv_matrix_dim = (num_heads, embedding_size, embedding_size // num_heads)

        # self attention layer
        self.q_matrix = nn.Parameter(torch.randn(*self.qkv_matrix_dim), requires_grad=True)
        self.k_matrix = nn.Parameter(torch.randn(*self.qkv_matrix_dim), requires_grad=True)
        self.v_matrix = nn.Parameter(torch.randn(*self.qkv_matrix_dim), requires_grad=True)

        self.feature_reduction_matrix = nn.Parameter(
            torch.randn(batch_size, embedding_size, self.dim_model), requires_grad=True)

        # feed forward layer
        self.feed_forward_layer = nn.Linear(self.dim_model, self.embedding_size)
        self.activation_function = nn.ReLU()

    def get_qkv(self, x, matrix):
        """qkv stands for q, k or v

        Following convention:
        - batch_size = 3
        - num_heads = 8
        - num_tokens = 32
        - embedding_size = 512
        - head_feature = embedding_size / num_heads = 64
        """

        qkv = torch.tensordot(x, matrix, dims=([2], [1]))  # 3, 32, 8, 64
        qkv = torch.swapaxes(qkv, 1, 2)  # 3, 8, 32, 64
        qkv = torch.reshape(qkv, (self.batch_size * self.num_heads, self.num_tokens, self.head_feature))  # 24, 32, 64

        return qkv  # 24, 32, 64

    def qkv_product(self, q, k, v, mask=None):
        """Takes q, k, v and returns z

        This is used both in self_attention and in encoder_decoder_attention.
        """
        kt = torch.transpose(k, 1, 2)  # 24, 64, 32
        z = torch.bmm(q, kt)  # 24, 32, 32

        if mask is not None:
            z = z.masked_fill_(mask == 0, -1e9)

        z = F.softmax(torch.div(z, self.head_feature ** 0.5), dim=-1)  # 24, 32, 32
        z = torch.bmm(z, v)  # 24, 32, 64
        z = torch.reshape(z, (self.batch_size, self.num_heads, self.num_tokens, self.head_feature))  # 3, 8, 32, 64
        z = torch.swapaxes(z, 1, 2)  # 3, 32, 8, 64
        z = torch.reshape(z, (self.batch_size, self.num_tokens, self.embedding_size))  # 3, 32, 512

        return z

    def self_attention(self, x, attention_mask):
        q = self.get_qkv(x, self.q_matrix)
        k = self.get_qkv(x, self.k_matrix)
        v = self.get_qkv(x, self.v_matrix)

        z = self.qkv_product(q, k, v, attention_mask)

        return torch.bmm(z, self.feature_reduction_matrix)  # 3, 32, 512

    def feed_forward(self, x):
        z = self.feed_forward_layer(x)
        return self.activation_function(z)

    def extra_repr(self) -> str:
        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])

        named_modules = list(named_modules)

        string_repr = ''
        for p in self.named_parameters():
            name = p[0].split('.')[0]
            if name not in named_modules:
                string_repr += f'({name}): tensor({str(tuple(p[1].shape))}, ' \
                               f'requires_grad={str(p[1].requires_grad)})\n'

        return string_repr
