from src.dataloader import train_dataloader
from src.device import DEVICE
from src.model import Transformer

from src.parameters import EMBEDDING_SIZE, POSITIONAL_ENCODING_SCALAR, NUM_TOKENS, BATCH_SIZE, NUM_HEADS, NUM_ENCODERS
from src.tokenizer import VOCABULARY_SIZE, tokenizer

if __name__ == '__main__':
    transformer = Transformer(
        vocabulary_size=VOCABULARY_SIZE,
        embedding_size=EMBEDDING_SIZE,
        num_tokens=NUM_TOKENS,
        positional_encoding_scalar=POSITIONAL_ENCODING_SCALAR,
        num_heads=NUM_HEADS,
        num_encoders=NUM_ENCODERS,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )

    # print(transformer)

    i, batch = next(enumerate(train_dataloader))

    # print(batch)

    output = transformer(batch)
    print("transformer output size:", output.size(), "\n")

    for original_sentence, translated_sentence in zip(
            tokenizer.batch_decode(batch['input_ids']), tokenizer.batch_decode(output)):
        print(original_sentence)
        print(translated_sentence)
        print()
