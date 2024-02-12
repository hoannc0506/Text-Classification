import os
import torch
from datasets import load_dataset
import evaluate
import numpy as np
import wandb

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
    )


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_name = "imdb"
raw_dataset = load_dataset(dataset_name)
print(len(raw_dataset['train']), len(raw_dataset['test']))

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


pretrain_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pretrain_name)

preprocess_funtion = lambda batch: tokenizer(batch['text'], 
                                             padding="max_length",
                                             truncation=True,
                                             return_tensors='pt')

tokenized_dataset = raw_dataset.map(preprocess_funtion, batched=True, num_proc=10)

model = AutoModelForSequenceClassification.from_pretrained(pretrain_name,
                                                           num_labels=2,
                                                           id2label=id2label,
                                                           label2id=label2id)

# prepare metric
metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return metric.compute(predictions=predictions, 
                          references=labels, 
                          average="weighted")


with wandb.init(project="text_classification") as run:
    save_dir = "results/imdb_distilbert_hf"
    # define training parameters
    training_args = TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=5e-5,
        num_train_epochs=10,
        # torch_compile=True, # optimizations
        optim="adamw_torch_fused", # optimizer
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
        run_name="imdb-distilbert"
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
    
