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
all_labels = []
all_predictions = []
with torch.no_grad():
    for idx, batch in enumerate(tqdm(dataloader)):
        # Do whatever processing you need
        input_ids, labels = batch["input_ids"], batch['labels']
        input_ids = input_ids.squeeze(1).to(device)
        outputs = model.generate(input_ids)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        all_predictions = all_predictions + [label2id[x] for x in predictions]
        all_labels = all_labels + labels.tolist()
          # Example usage

    f1_score = metric.compute(references=all_labels, 
                              predictions=all_predictions,
                              average="weighted")

    print(f1_score)
