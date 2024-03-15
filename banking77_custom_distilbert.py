import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
sys.path.insert(0, os.path.dirname('utils'))
sys.path.insert(0, os.path.dirname('models'))

from models.pretrained import TextClassifier
from utils import trainer
import wandb
from datasets import load_dataset

dataset_name = "banking77"
raw_dataset = load_dataset(dataset_name)

print(len(raw_dataset['train']), len(raw_dataset['test']))

model = TextClassifier(pretrained_name="distilbert-base-uncased")

preprocess_function = lambda sample: model.tokenizer(sample['text'],
                                                     padding="max_length",
                                                     truncation=True,
                                                     return_tensors='pt')

tokenized_dataset = raw_dataset.map(preprocess_function, 
                                    batched=True)

print(len(tokenized_dataset['train']), len(tokenized_dataset['test']))
# Define batch size
batch_size = 32

# Create PyTorch DataLoaders for training and validation
train_loader = DataLoader(tokenized_dataset['train'], batch_size=batch_size, shuffle=True)
val_loader = DataLoader(tokenized_dataset['test'], batch_size=batch_size, shuffle=False)


# training
device = 'cuda' if torch.cuda.is_available else 'cpu'
model = model.to(device)

epochs = 10
LR = 5e-5
criterion = nn.CrossEntropyLoss()
scheduler_step_size = epochs *0.6
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)

wandb_logger = wandb.init(project="Text-classification",
                          name="Transformer_basic_tokenizer",
                          config={'save_dir':"results/banking77_custom_distilbert"})


train_losses, val_losses = trainer.train(model, 
                                         train_loader, 
                                         val_loader, 
                                         criterion, 
                                         optimizer, 
                                         scheduler=None, 
                                         device=device,
                                         epochs=epochs,
                                         logger=wandb_logger)

