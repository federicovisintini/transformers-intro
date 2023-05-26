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
            batch_size: int,
            device: str | torch.device
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.hidden_dim_final_layer = 512  # this is arbitrary, does not need to be embedding_size

        self.batch_size = batch_size
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
            batch_size=batch_size
        )

        self.encoders = nn.ModuleList([
            Encoder(
                embedding_size=embedding_size,
                num_tokens=num_tokens,
                batch_size=batch_size,
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
                batch_size=batch_size,
                num_heads=num_heads
            ) for _ in range(num_encoders)
        ])

        # final layer
        self.feed_forward_layer1 = nn.Linear(embedding_size * num_tokens, self.hidden_dim_final_layer)
        self.activation_function = nn.ReLU()
        self.feed_forward_layer2 = nn.Linear(self.hidden_dim_final_layer, vocabulary_size)
        self.softmax = nn.Softmax(dim=1)

        self.to(self.device)

    def final_layer(self, x):
        x = torch.flatten(x, start_dim=1)

        z = self.feed_forward_layer1(x)
        x1 = self.activation_function(z)

        z1 = self.feed_forward_layer2(x1)
        return self.softmax(z1)

    def forward(self, batch):
        input_token_ids = batch['input_ids']
        input_attention_mask = batch['input_attention_mask']

        # encoder side
        embedded_tokens = self.embedding(input_token_ids)
        x = self.positional_encoder(embedded_tokens)

        for i, encoder in enumerate(self.encoders):
            x = encoder(x, input_attention_mask)  # 3, 32, 512

        k = self.encoders[-1].get_qkv(x, self.k_matrix)
        v = self.encoders[-1].get_qkv(x, self.v_matrix)

        # decoder side
        # TODO is attention mask on past values correct?
        # TODO transformer should stop when generates [SEP]
        output_tokens = torch.zeros(self.batch_size, self.num_tokens, dtype=torch.int64).to(self.device)
        decoder_attention_mask = torch.zeros(self.batch_size * self.num_heads, self.num_tokens, self.num_tokens)

        for j in range(self.num_tokens):
            embedded_tokens = self.embedding(output_tokens)
            x = self.positional_encoder(embedded_tokens)
            decoder_attention_mask[:, :j, :j] = 1

            for i, decoder in enumerate(self.decoders):
                x = decoder(x, k, v, input_attention_mask, decoder_attention_mask)

            z = self.final_layer(x)
            tokens = torch.argmax(z.to("cpu"), dim=1)  # one token per batch

            output_tokens[:, j] = tokens

        return output_tokens
