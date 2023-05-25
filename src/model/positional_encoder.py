import torch
from torch import nn

from src.parameters import POSITIONAL_ENCODING_COEFFICIENTS


class PositionalEncoder(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, num_tokens, positional_encoding_scalar, batch_size):
        super().__init__()

        assert embedding_size % 2 == 0

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.positional_encoding_scalar = positional_encoding_scalar
        self.batch_size = batch_size

        self.positional_encoding_vector = nn.Parameter(
            self.compute_positional_encoding_vector(), requires_grad=False)

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
        return embedded_token + self.positional_encoding_vector / POSITIONAL_ENCODING_COEFFICIENTS

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
