import torch
from torch import nn

from src.model.decoder import Decoder
from src.model.embedding import Embedding
from src.model.encoder import Encoder
from src.model.positional_encoder import PositionalEncoder


class Transformer(nn.Module):
    def __init__(
            self,
            vocabulary_size: int,
            embedding_size: int,
            num_tokens: int,
            positional_encoding_scalar: int | float,
            num_heads: int,
            num_encoders: int,
            device: str | torch.device
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.device = device

        self.embedding = Embedding(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size
        )

        self.positional_encoder = PositionalEncoder(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            num_tokens=num_tokens,
            positional_encoding_scalar=positional_encoding_scalar,
        )

        self.encoders = nn.ModuleList([
            Encoder(
                embedding_size=embedding_size,
                num_tokens=num_tokens,
                num_heads=num_heads
            ) for _ in range(num_encoders)
        ])

        # encoder-decoder-attention
        qkv_matrix_dim = (num_heads, embedding_size, embedding_size // num_heads)
        self.k_matrix = nn.Parameter(torch.randn(*qkv_matrix_dim), requires_grad=True)
        self.v_matrix = nn.Parameter(torch.randn(*qkv_matrix_dim), requires_grad=True)

        self.decoders = nn.ModuleList([
            Decoder(
                embedding_size=embedding_size,
                num_tokens=num_tokens,
                num_heads=num_heads
            ) for _ in range(num_encoders)
        ])

        # final layer
        self.feed_forward_layer = nn.Linear(embedding_size, vocabulary_size)
        self.softmax = nn.Softmax(dim=-1)

        self.to(self.device)

    def final_layer(self, x):
        z = self.feed_forward_layer(x)
        return self.softmax(z)

    def forward(self, batch):
        # encoder side
        input_token_ids = batch['input_ids']
        input_attention_mask = batch['input_attention_mask']

        embedded_tokens = self.embedding(input_token_ids)
        x = self.positional_encoder(embedded_tokens)

        for i, encoder in enumerate(self.encoders):
            x = encoder(x, input_attention_mask)  # 3, 32, 512

        k = self.encoders[-1].get_qkv(x, self.k_matrix)
        v = self.encoders[-1].get_qkv(x, self.v_matrix)

        # decoder side

        # TRAINING
        # instead of passing recursively the predicted output,
        # we pass as decoder-input the true labels and ask to infer the next word
        # this leads to more stability during training and allows for parallel training
        # TODO shift decoder input
        output_token_ids = batch['output_ids']
        output_attention_mask = batch['input_attention_mask']
        batch_size = len(output_token_ids)

        # triangular_lower: q x kt; the first word only knows itself; the last word knows all of them
        future_attention_mask = torch.tril(
            torch.ones(batch_size * self.num_heads, self.num_tokens, self.num_tokens, requires_grad=False)
        )

        decoder_attention_mask = output_attention_mask * future_attention_mask

        embedded_tokens = self.embedding(output_token_ids)
        x = self.positional_encoder(embedded_tokens)

        for i, decoder in enumerate(self.decoders):
            x = decoder(x, k, v, input_attention_mask, decoder_attention_mask)

        probas = self.final_layer(x)

        return probas
