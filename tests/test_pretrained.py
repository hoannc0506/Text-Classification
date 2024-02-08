#%% 
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

#%%
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# %%
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                           num_labels=2,
                                                           id2label=id2label,
                                                           label2id=label2id)
# %%
import torchinfo
config = model.config
input_ids = torch.randint(0, config.vocab_size, size=(1, 300))
torchinfo.summary(model, input_data=input_ids)

# %%
sample = "I am student at Ho Chi Minh City University of Technology"

tokens = tokenizer(sample)
# %%
