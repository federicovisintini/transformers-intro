from src.dataloader import train_dataloader, VOCABULARY_SIZE
from src.device import DEVICE
from src.model.encoder import Encoder
from src.model.positional_encoder import PositionalEncoder
from src.model.embedding import Embedding
from src.model.transformer import Transformer
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
    encoders = [
        Encoder(
            embedding_size=EMBEDDING_SIZE,
            num_tokens=NUM_TOKENS,
            batch_size=BATCH_SIZE,
            num_heads=NUM_HEADS
        ).to(DEVICE) for _ in range(6)
    ]

    tranformer = Transformer(embedding, pos_encoder, encoders)

    print(tranformer)

    i, batch = next(enumerate(train_dataloader))

    z = tranformer(batch)
    print()
    print(z.size())

    # embedded_tokens = embedding(batch)
    # print(embedded_tokens.size())  # 3, 32, 512
    #
    # positional_encoded_tokens = pos_encoder(embedded_tokens)
    # print(positional_encoded_tokens.size())  # 3, 32, 512
    #
    # z = encoder(positional_encoded_tokens)
    # print(z.size())
