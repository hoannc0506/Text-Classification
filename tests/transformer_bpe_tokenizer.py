import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.path.dirname('utils'))

from utils.vocab_builders import build_bpe_vocab
from torch.utils.data import DataLoader
from utils.dataset_builders import IMDB_Dataset
from datasets import load_dataset
from models.transformer import TransformerTextCls
from utils import trainer
import wandb

# Correct dataset name for IMDb is "imdb"
imdb = load_dataset("imdb")
train_data, test_data = imdb['train'], imdb['test']
print(next(iter(train_data)))

min_freq = 3
vocab_size = 20000
bpe_vocab, bpe_tokenizer = build_bpe_vocab(train_data, min_freq, vocab_size)

bpe_text_pipeline = lambda x: bpe_vocab(bpe_tokenizer.encode(x).tokens)

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
seq_length = 200
batch_size = 32

# using torch dataset
train_dataset = IMDB_Dataset(train_data, bpe_vocab, bpe_text_pipeline, seq_length=seq_length)
val_dataset = IMDB_Dataset(test_data, bpe_vocab, bpe_text_pipeline, seq_length=seq_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = TransformerTextCls(vocab_size, 
                           max_length=seq_length,
                           num_classes=2,
                           embed_dim=32,
                           num_heads=2,
                           ff_dim=32,
                           dropout=0.1, 
                           device=device)

model = model.to(device)

epochs = 20
LR = 0.001
criterion = nn.CrossEntropyLoss()
scheduler_step_size = epochs *0.6
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)

wandb_logger = wandb.init(project="Text-classification",
                          name="Transformer_bpe_tokenizer",
                          config={'save_dir':"results"})


train_losses, val_losses = trainer.train(model, 
                                         train_loader, 
                                         test_loader, 
                                         criterion, 
                                         optimizer, 
                                         scheduler=None, 
                                         device=device,
                                         epochs=epochs,
                                         logger=wandb_logger)
