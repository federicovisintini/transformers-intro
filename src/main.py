import math

import torch
import torch.nn.functional as F

from src.dataloader import train_dataloader, VOCABULARY_SIZE
from src.device import DEVICE
from src.model.encoder import Encoder
from src.model.positional_encoding import PositionalEncoder
from src.model.embedding import Embedding
from src.parameters import EMBEDDING_SIZE, POSITIONAL_ENCODING_SCALAR, NUM_TOKENS, BATCH_SIZE, NUM_HEADS

if __name__ == '__main__':
    # embedding
    embedding = Embedding(vocabulary_size=VOCABULARY_SIZE, embedding_size=EMBEDDING_SIZE)
    embedding.to(DEVICE)

    # positional encoder
    pos_encoder = PositionalEncoder(
        vocabulary_size=VOCABULARY_SIZE,
        embedding_size=EMBEDDING_SIZE,
        num_tokens=NUM_TOKENS,
        positional_encoding_scalar=POSITIONAL_ENCODING_SCALAR,
        batch_size=BATCH_SIZE
    )
    pos_encoder.to(DEVICE)

    # encoder
    encoder = Encoder(
        embedding_size=EMBEDDING_SIZE,
        num_tokens=NUM_TOKENS,
        batch_size=BATCH_SIZE,
        num_heads=NUM_HEADS
    )
    encoder.to(DEVICE)

    i, batch = next(enumerate(train_dataloader))

    embedded_tokens = embedding(batch)
    print(embedded_tokens.size())  # 3, 32, 512

    positional_encoded_tokens = pos_encoder(embedded_tokens)
    print(positional_encoded_tokens.size())  # 3, 32, 512

    z = encoder(positional_encoded_tokens)
    print(z.size())

    # # test
    # attention_matrix = torch.randn(8, 512, 64).to(DEVICE)
    # q = torch.tensordot(positional_encoded_tokens, attention_matrix, dims=([2], [1]))  # 3, 32, 8, 64
    # q = torch.swapaxes(q, 1, 2)
    # q = k = v = torch.reshape(q, (24, 32, 64))  # 24, 32, 64
    #
    # kt = torch.transpose(k, 1, 2)
    # z = torch.bmm(q, kt)
    # z = F.softmax(torch.div(z, math.sqrt(64)), dim=2)  # 24, 32, 32
    # z = torch.bmm(z, v)  # 24, 32, 64
    # z = torch.reshape(z, (3, 8, 32, 64))
    # z = torch.swapaxes(z, 1, 2)
    # z = torch.reshape(q, (3, 32, 512))
    #
    # feature_reduction_matrix = torch.randn(3, 512, 512).to(DEVICE)
    # z = torch.tensordot(z, feature_reduction_matrix)
    #
    # print(z.size())
