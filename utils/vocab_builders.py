from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchtext.data.utils import get_tokenizer


def yield_tokens(data_iter, tokenizer):
    for data in data_iter:
        yield tokenizer(data)

def build_basic_vocab(data, min_freq, vocab_size):
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(yield_tokens(data, tokenizer),
                                  min_freq=min_freq,
                                  max_tokens=vocab_size,
                                  specials=["<pad>", "<s>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    
    return vocab, tokenizer




