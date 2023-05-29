import torch
from tqdm import tqdm

from src.dataloader import train_dataloader
from src.dataloader import VOCABULARY_SIZE, tokenizer
from src.model import Transformer
from src.parameters import EMBEDDING_SIZE, POSITIONAL_ENCODING_SCALAR, NUM_TOKENS, NUM_HEADS, NUM_ENCODERS
from src.utils import count_parameters
from src.utils import DEVICE


# PREDICTIONS
def predict(model, batch):
    cls_id, sep_id = tokenizer('')['input_ids']

    output = [torch.tensor([cls_id], requires_grad=False)]
    for token_number in tqdm(range(model.num_tokens - 1)):
        # decoder_attention_mask = torch.zeros(
        #   self.batch_size * self.num_heads, self.num_tokens, self.num_tokens)
        # TODO set batch['output_ids'], batch['output_attention_mask']

        with torch.no_grad():
            probas = model(batch)

        tokens = torch.argmax(probas.to("cpu"), dim=-1)
        next_token = tokens[:, token_number]
        output.append(next_token)

        if next_token.item() == sep_id:
            break

    output = torch.stack(output)
    return torch.swapaxes(output, 0, 1)


if __name__ == '__main__':
    transformer = Transformer(
        vocabulary_size=VOCABULARY_SIZE,
        embedding_size=EMBEDDING_SIZE,
        num_tokens=NUM_TOKENS,
        positional_encoding_scalar=POSITIONAL_ENCODING_SCALAR,
        num_heads=NUM_HEADS,
        num_encoders=NUM_ENCODERS,
        device=DEVICE
    )

    # print(transformer)
    print(f"{count_parameters(transformer) / 1e6:.0f}M parameters")

    i, batch = next(enumerate(train_dataloader))

    for key, value in batch.items():
        batch[key] = value[:1]

    tokens = predict(transformer, batch)

    print("transformer output tokens size:", tokens.size(), "\n")

    for original_sentence, translated_sentence in zip(
            tokenizer.batch_decode(batch['input_ids']), tokenizer.batch_decode(tokens)):
        print(original_sentence)
        print(translated_sentence)
        print()
