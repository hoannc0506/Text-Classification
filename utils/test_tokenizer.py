#%%
import torch 
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

#%%
# Sample text data
text_data = [
    "This is the first sentence.",
    "Another sentence for tokenization.",
    "PyTorch is awesome!",
]

tokenizer = get_tokenizer("basic_english")
basic_tokenized_data = [tokenizer(text) for  text in text_data]

vocab = build_vocab_from_iterator(basic_tokenized_data, 
                                  specials=["<unk>", "<pad>"])

#%%
# Print vocabulary information
print("Vocabulary size:", len(vocab))
print("Vocabulary tokens:", vocab.get_itos())
print("Numerical indices:", vocab.get_stoi())
# %%

# %%
import spacy

spacy_tokenizer = spacy.load('en_core_web_sm')

#%%
def spacy_tokenize(text):
    return [token.text for token in spacy_tokenizer(text)]

# spacy_tokenize = lambda texts: [token.text for token in spacy_tokenize(texts)]

# %%
spacy_tokenized_data = [spacy_tokenize(sequence) for sequence in text_data]

# %%
from tokenizers import Tokenizer
from tokenizers.models import BPE

from tokenizers.pre_tokenizers import Whitespace


tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

# %%
tokenized_data = [tokenizer.encode(text).tokens for text in text_data]

# %%
from tokenizers.trainers import BpeTrainer
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"], trainer=trainer)

# %%
