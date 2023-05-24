from src.dataloader import train_dataloader, VOCABULARY_SIZE
from src.device import DEVICE
from src.model import init_transformer

from src.parameters import EMBEDDING_SIZE, POSITIONAL_ENCODING_SCALAR, NUM_TOKENS, BATCH_SIZE, NUM_HEADS

if __name__ == '__main__':
    transformer = init_transformer(
        vocabulary_size=VOCABULARY_SIZE,
        embedding_size=EMBEDDING_SIZE,
        num_tokens=NUM_TOKENS,
        positional_encoding_scalar=POSITIONAL_ENCODING_SCALAR,
        num_heads=NUM_HEADS,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )

    print(transformer)

    i, batch = next(enumerate(train_dataloader))

    z = transformer(batch)

    print(z.size())
