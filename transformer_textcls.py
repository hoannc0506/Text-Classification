# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
nltk.download('punkt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
from datasets import load_dataset

# Correct dataset name for IMDb is "imdb"
imdb = load_dataset("imdb")
train_data, test_data = imdb['train'], imdb['test']
print(next(iter(train_data)))

# %%
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# %%
tokenizer = get_tokenizer("basic_english")
vocab_size = 20000

def yield_tokens(data_iter):
    for data in data_iter:
        yield tokenizer(data["text"])
        
# %%
vocab = build_vocab_from_iterator(yield_tokens(train_data),
                                  min_freq=3,
                                  max_tokens=vocab_size,
                                  specials=["<pad>", "<s>", "<unk>"])

vocab.set_default_index(vocab["<unk>"])

# %%
seq_length = 200
text_pipeline = lambda x: vocab(tokenizer(x))

# %%
def collate_batch(batch, seq_length=seq_length):
    text_list, label_list = [], []
    for sample in batch:
        label_list.append(sample['label'])
        text_processed = text_pipeline(sample['text'])[:seq_length]
        
        if len(text_processed) < seq_length:
            pad_size = seq_length - len(text_processed) -1
            # pad before
            text_processed = [vocab["<pad>"]]*pad_size + [vocab["<s>"]] + text_processed
            
        input_ids = torch.tensor(text_list, dtype=torch.int64)
        labels = torch.tensor(label_list, dtype=torch.int64)
        
        return (input_ids, labels)
    
        
# %%
from torch.utils.data import DataLoader
batch_size = 32

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, drop_last=True)
# %%

from models.transformer import TransformerTextCls

model = TransformerTextCls(vocab_size, 
                           max_length=seq_length,
                           num_classes=2,
                           embed_dim=32,
                           num_heads=2,
                           ff_dim=32,
                           dropout=0.1, 
                           device=device)

max_epoch = 20
LR = 0.001
criterion = nn.CrossEntropyLoss()
scheduler_step_size = epochs *0.6
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)

# %%
import trainer
import wandb

wandb_logger = wandb.init(project="Text-classification",
                          name="Transformer_basic_tokenizer")
# %%
train_losses, val_losses = trainer.train(model, train_loader, test_loader, 
                                         criterion, optimizer, scheduler, 
                                         device, wandb_logger)
