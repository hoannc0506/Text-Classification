import torch
from torch.utils.data import Dataset

class IMDB_Dataset(Dataset):
    def __init__(self, data, vocab, tokenizer, seq_length=30):
        self.data = data
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        # text pipeline
        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data[index].get('text')
        label = self.data[index].get('label')
        
        tokens = self.text_pipeline(text)[:self.seq_length]
        # padding 
        if len(tokens) < self.seq_length:
            pad_size = self.seq_length - len(tokens) -1
            # pad before
            tokens = [self.vocab["<pad>"]]*pad_size + [self.vocab["<s>"]] + tokens
            
            # pad after
            # sequence += [self.vocab['<pad>']] * (self.max_sequence_length - len(sequence))

        tokens = torch.tensor(tokens, dtype=torch.int64)
        labels = torch.tensor(label, dtype=torch.int64)
    
        return (tokens, labels)
        
class Tweets_Dataset(Dataset):
    pass

if __name__ == '__main__':
    from datasets import load_dataset
    from vocab_builders import build_basic_vocab

    # Correct dataset name for IMDb is "imdb"
    imdb = load_dataset("imdb")
    train_data, test_data = imdb['train'], imdb['test']
    print(next(iter(train_data)))
    
    min_freq = 3
    vocab_size = 20000
    vocab, basic_tokenizer = build_basic_vocab(train_data, min_freq, vocab_size)
    
    sample_dataset = IMDB_Dataset(train_data, vocab, basic_tokenizer, 
                                  seq_length=200)
    
    print(next(iter(sample_dataset)))