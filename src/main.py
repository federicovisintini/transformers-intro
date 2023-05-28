import torch

from src.dataloader.dataloader import train_dataloader
from src.dataloader.tokenizer import VOCABULARY_SIZE, tokenizer
from src.model import Transformer
from src.parameters import EMBEDDING_SIZE, POSITIONAL_ENCODING_SCALAR, NUM_TOKENS, BATCH_SIZE, NUM_HEADS, NUM_ENCODERS
from src.utils.device import DEVICE

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

    probas = transformer(batch)
    tokens = torch.argmax(probas.to("cpu"), dim=-1)

    print("transformer output tokens size:", tokens.size(), "\n")

    for original_sentence, translated_sentence in zip(
            tokenizer.batch_decode(batch['input_ids']), tokenizer.batch_decode(tokens)):
        print(original_sentence)
        print(translated_sentence)
        print()
