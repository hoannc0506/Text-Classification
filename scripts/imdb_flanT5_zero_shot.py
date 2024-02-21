from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import torch
import numpy as np
import evaluate

# load dataset
dataset = load_dataset("imdb", split="test")
id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

# load model
device = "cuda:1"
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map=device)

# preprocess function
instruct_promt = "Please classify the sentiment of the following statement as 'positive' or 'negative'. Statement: "

# add instruct prompt to dataset
# dataset = dataset.map(lambda example: {'text': instruct_promt + example['text']})

tokenize_function = lambda batch: tokenizer(
    instruct_promt + batch["text"], 
    padding="max_length", 
    truncation=True,
    return_tensors='pt'
)

# tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, num_proc=20)
tokenized_dataset = tokenized_dataset.remove_columns('text')
tokenized_dataset

# import pdb; pdb.set_trace()

# # Convert to Hugging Face Dataset format
# tokenized_dataset = Dataset.from_dict({
#     'input_ids': tokenized_dataset['input_ids'],
#     'label': tokenized_dataset['label'],
# })

# Create DataCollator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define batch size
batch_size = 64

# Create DataLoader
dataloader = DataLoader(
    tokenized_dataset, 
    batch_size=batch_size, 
    collate_fn=data_collator,
    shuffle=False
)

metric = evaluate.load("f1")
model.eval()

# Iterate through DataLoader
with torch.no_grad():
    for batch in dataloader:
        import pdb;pdb.set_trace()
        input_ids = batch['input_ids']
        labels = batch['labels']
        # Do whatever processing you need
        print(input_ids.shape, labels.shape)  # Example usage

        break