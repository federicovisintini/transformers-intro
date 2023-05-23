from src.dataloader import train_dataloader, VOCABULARY_SIZE
from src.device import DEVICE
from src.model.positional_encoding import PositionalEncoder
from src.model.embedding import Embedding
from src.parameters import EMBEDDING_SIZE, POSITIONAL_ENCODING_SCALAR, NUM_TOKENS, BATCH_SIZE

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

    i, batch = next(enumerate(train_dataloader))

    embedded_tokens = embedding(batch)
    print(embedded_tokens)  # 3, 32, 512

    positional_encoded_tokens = pos_encoder(embedded_tokens)
    print(positional_encoded_tokens)  # 3, 32, 512
