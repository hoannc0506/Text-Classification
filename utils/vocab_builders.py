from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchtext.data.utils import get_tokenizer
from tokenizer import Tokenizer
from tokenizer.models import BPE
from tokenizer.pre_tokenizers import Whitespace

def yield_tokens(data_iter, tokenizer):
    for data in data_iter:
        yield tokenizer(data['text'])

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
    
    tokenized_data = 




