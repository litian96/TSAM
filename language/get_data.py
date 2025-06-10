from datasets import load_dataset



def get_data(dataset_name, tokenizer):

    def tokenize(ex):
        #tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        if dataset_name in ['wnli', 'stsb', 'mrpc', 'rte']:
            encoded_dict = tokenizer.encode_plus(
                ex['sentence1'],
                ex['sentence2'],
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=64,            # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,   # Construct attn. masks.
                return_tensors='pt',          # Return pytorch tensors.
            )
        elif dataset_name in ['sst2', 'cola']:
            encoded_dict = tokenizer.encode_plus(
                ex['sentence'],
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=64,            # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,   # Construct attn. masks.
                return_tensors='pt',          # Return pytorch tensors.
            )
        elif dataset_name == 'mnli':
            encoded_dict = tokenizer.encode_plus(
                ex['premise'],
                ex['hypothesis'],
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=64,            # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,   # Construct attn. masks.
                return_tensors='pt',          # Return pytorch tensors.
            )
        elif dataset_name == 'qnli':
            encoded_dict = tokenizer.encode_plus(
                ex['question'],
                ex['sentence'],
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=64,            # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,   # Construct attn. masks.
                return_tensors='pt',          # Return pytorch tensors.
            )
        elif dataset_name == 'qqp':
            encoded_dict = tokenizer.encode_plus(
                ex['question1'],
                ex['question2'],
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=64,            # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,   # Construct attn. masks.
                return_tensors='pt',          # Return pytorch tensors.
            )
        encoded_dict['label'] = ex['label']
        if dataset_name == 'stsb':
            encoded_dict['label'] = ex['label'] / 5  # normalize to [0,1]
        return encoded_dict


    dataset = load_dataset("nyu-mll/glue", dataset_name, cache_dir='')
    train_ds = dataset["train"]
    if dataset_name == "mnli":
        test_ds = dataset["validation_matched"]  # dev set
    else:
        test_ds = dataset["validation"]

    train_ds, test_ds = train_ds.map(tokenize), test_ds.map(tokenize)

    return train_ds, test_ds

