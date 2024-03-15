import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import evaluate
import numpy as np
import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_name = "banking77"
raw_dataset = load_dataset(dataset_name)
print(len(raw_dataset['train']), len(raw_dataset['test']))


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize helper function
tokenize = lambda batch: tokenizer(batch['text'], 
                                   padding="max_length",
                                   truncation=True,
                                   return_tensors='pt')
# Tokenize dataset
raw_dataset = raw_dataset.rename_column("label", "labels") #match trainer column
tokenized_dataset = raw_dataset.map(tokenize, batched=True, remove_columns=["text"])

print(len(tokenized_dataset['train']), len(tokenized_dataset['test']))

# prepare model, labels
labels = tokenized_dataset['train'].features['labels'].names
label2id = {label:idx for idx, label in enumerate(labels)}  
id2label = {idx:label for idx, label in enumerate(labels)}

model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                           num_labels=len(labels),
                                                           label2id=label2id,
                                                           id2label=id2label)

model.to(device)

# prepare metric
metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")


with wandb.init(project="text_classification") as run:
    save_dir = "results/banking77_distilbert"
    # define training parameters
    training_args = TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=5e-5,
        num_train_epochs=10,
        # torch_compile=True, # optimizations
        optim="adamw_hf", # optimizer
        # logging & evaluation strategies
        logging_dir=f"{save_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        # wandb logging
        report_to="wandb",
        run_name="banking77-distilbert"
    )

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    