from torchtext.vocab import build_vocab_from_iterator, vocab, Vocab
from torchtext.data.utils import get_tokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def yield_tokens(data_iter, tokenizer):
    for data in data_iter:
        yield tokenizer(data['text'])

def imdb_data_iter(data):
    for sample in data:
        yield sample['text']

def build_basic_vocab(data, min_freq, vocab_size):
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(yield_tokens(data, tokenizer),
                                  min_freq=min_freq,
                                  max_tokens=vocab_size,
                                  specials=["<pad>", "<s>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    
    return vocab, tokenizer


def build_bpe_vocab(data, min_freq, vocab_size):
    bpe_tokenizer = Tokenizer(BPE())
    bpe_tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size,
                         min_frequency=min_freq,
                         special_tokens=["<pad>", "<s>", "<unk>"])
    
    bpe_tokenizer.train_from_iterator(iterator=imdb_data_iter(data),
                                      trainer=trainer,
                                      length=len(data))
    
    bpe_vocab = vocab(ordered_dict=bpe_tokenizer.get_vocab(),
                      min_freq=min_freq, 
                      specials=["<pad>", "<s>", "<unk>"])

    bpe_vocab.set_default_index(bpe_vocab["<unk>"])
    
    return bpe_vocab, bpe_tokenizer

if __name__ == "__main__":
    from datasets import load_dataset
    imdb = load_dataset("imdb")
    train_data, test_data = imdb['train'], imdb['test']
    
    test_vocab, test_tokenizer = build_bpe_vocab(train_data, min_freq=3, vocab_size=20000)
    
    import pdb; pdb.set_trace()
    print('hear')

    
    


