from src.dataloader import tokenizer, train_dataloader, VOCABULARY_SIZE
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

    output = transformer(batch)
    print("transformer output size:", output.size(), "\n")

    for original_sentence, translated_sentence in zip(
            tokenizer.batch_decode(batch['input_ids']), tokenizer.batch_decode(output)):
        print(original_sentence)
        print(translated_sentence)
        print()
