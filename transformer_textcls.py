# %%
from datasets import load_dataset

imdb = load_dataset("imdb")
imdb.save_to_disk("dataset")
train_data, test_data = imdb['train'], imdb['test']