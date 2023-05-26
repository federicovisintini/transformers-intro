from datasets import load_dataset
from torch.utils.data import DataLoader

from src.device import DEVICE
from src.parameters import BATCH_SIZE, NUM_TOKENS
from src.tokenizer import tokenizer


def input_tokenization(batch, num_tokens=NUM_TOKENS):
    # process_data_to_model_inputs
    inputs = tokenizer(
        text=[segment['translation']['en'] for segment in batch],
        padding="max_length",
        truncation=True,
        max_length=num_tokens,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = tokenizer(
        [segment['translation']['de'] for segment in batch],
        padding="max_length",
        truncation=True,
        max_length=num_tokens,
        return_tensors="pt"
    ).to(DEVICE)

    batch = {
        "input_ids": inputs.input_ids,
        "input_attention_mask": inputs.attention_mask,
        "output_ids": outputs.input_ids,
        "output_attention_mask": outputs.attention_mask,
        "labels": outputs.input_ids
    }

    # because BERT automatically shifts the labels,
    # the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels]
                       for labels in batch["labels"]]

    return batch


train_ds = load_dataset("wmt14", 'de-en', split="train").with_format("torch")
validation_ds = load_dataset("wmt14", 'de-en', split="validation").with_format("torch")

train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=input_tokenization)
validation_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=input_tokenization)

if __name__ == '__main__':
    original_sentences = [train_ds[i]['translation']['en'] for i in range(BATCH_SIZE)]

    # decoded sentence
    i, batch = next(enumerate(train_dataloader))
    decoded_sentences = tokenizer.batch_decode(batch['input_ids'])

    for original_sentence, decoded_sentence in zip(original_sentences, decoded_sentences):
        print("original:", original_sentence)
        print("decoded: ", decoded_sentence)
        print()

    print(batch)
