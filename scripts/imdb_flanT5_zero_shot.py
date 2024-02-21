from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from tqdm import tqdm
import torch
import numpy as np
import evaluate

# load dataset
dataset = load_dataset("imdb", split="test")
id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

# load model
device = "cuda:2"
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

# Create DataCollator
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create DataLoader
dataloader = DataLoader(
    tokenized_dataset, 
    batch_size=64, 
    # collate_fn=data_collator,
    shuffle=True
)

# init metric
metric = evaluate.load("f1")
model.eval()
with torch.no_grad():
    for idx, batch in enumerate(tqdm(dataloader)):
        input_ids = torch.tensor(batch['input_ids']
        
        attention_mask = batch['attention_mask']
        # Do whatever processing you need
        print(input_ids.shape, attention_mask.shape)

        break

    