import torch

from src.device import DEVICE
from src.model.decoder import Decoder
from src.model.embedding import Embedding
from src.model.encoder import Encoder
from src.model.positional_encoder import PositionalEncoder
from src.model.transformer import Transformer


def init_transformer(
        vocabulary_size: int,
        embedding_size: int,
        num_tokens: int,
        positional_encoding_scalar: int | float,
        num_heads: int,
        batch_size: int,
        device: str | torch.device
):
    # embedding
    embedding = Embedding(vocabulary_size=vocabulary_size, embedding_size=embedding_size)
    embedding.to(device)

    # positional encoder
    pos_encoder = PositionalEncoder(
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size,
        num_tokens=num_tokens,
        positional_encoding_scalar=positional_encoding_scalar,
        batch_size=batch_size
    )
    pos_encoder.to(device)

    # encoder
    encoders = [
        Encoder(
            embedding_size=embedding_size,
            num_tokens=num_tokens,
            batch_size=batch_size,
            num_heads=num_heads
        ).to(DEVICE) for _ in range(6)
    ]

    # decoder
    decoders = [
        Decoder(
            embedding_size=embedding_size,
            num_tokens=num_tokens,
            batch_size=batch_size,
            num_heads=num_heads
        ).to(DEVICE) for _ in range(6)
    ]

    # transformer
    transformer = Transformer(
        embedding=embedding,
        positional_encoder=pos_encoder,
        encoders=encoders,
        decoders=decoders,
        embedding_size=embedding_size,
        num_tokens=num_tokens,
        vocabulary_size=vocabulary_size
    )

    transformer.to(device)

    return transformer
