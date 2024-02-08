import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from datasets import load_dataset

imdb = load_dataset("imdb")

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


from transformers import AutoTokenizer

pretrain_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pretrain_name)

def preprocess_funtion(examples):
    return tokenizer(examples['text'], truncation=True)

tokenized_imdb = imdb.map(preprocess_funtion, batched=True, num_proc=10)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(pretrain_name,
                                                           num_labels=2,,
                                                           id2label=id2label,
                                                           label2id=label2id)

import numpy as np
import evaluate

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, labels=labels)


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="results",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=20,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb['train'],
    eval_dataset=tokenized_imdb['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


trainer.evaluate(tokenized_imdb["test"])

