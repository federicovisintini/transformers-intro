from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-multilingual-cased")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.add_special_tokens({'bos_token': '[BOS]'})

VOCABULARY_SIZE = tokenizer.vocab_size + 1  # (BOS token)
