from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-multilingual-cased")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

VOCABULARY_SIZE = tokenizer.vocab_size
