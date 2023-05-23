from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

train_ds = load_dataset("wmt14", 'de-en', split="train").with_format("torch")
validation_ds = load_dataset("wmt14", 'de-en', split="validation").with_format("torch")

tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-multilingual-cased")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


def process_data_to_model_inputs(batch, num_tokens=32):
    inputs = tokenizer([segment['translation']['en'] for segment in batch],
                       padding="max_length", truncation=True, max_length=num_tokens)
    outputs = tokenizer([segment['translation']['de'] for segment in batch],
                        padding="max_length", truncation=True, max_length=num_tokens)

    batch = {}

    batch["input_ids"] = inputs.input_ids
    batch["input_attention_mask"] = inputs.attention_mask
    batch["output_ids"] = outputs.input_ids
    batch["output_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels,
    # the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels]
                       for labels in batch["labels"]]
    return batch


train_dataloader = DataLoader(train_ds, batch_size=3, shuffle=False, collate_fn=process_data_to_model_inputs)
validation_dataloader = DataLoader(train_ds, batch_size=3, shuffle=False, collate_fn=process_data_to_model_inputs)

for batch in train_dataloader:
    print(batch)
    break
