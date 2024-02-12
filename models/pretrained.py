#%%
import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel

class TextClassifier(nn.Module):
    def __init__(self, 
                 pretrained_name="distilbert-base-uncased",
                 num_classes=2,
                 dropout_prob=0.2):
        super(TextClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.config = self.model.config
        
        print("freezing pretrained parameters")
        for n, p in self.model.named_parameters():
            p.requires_grad = False
        
        fc_hidden_dim = int(0.5 * self.config.dim)
        
        self.fc1 = nn.Linear(self.config.dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        output = self.model(x)
        # import pdb;pdb.set_trace()
        x = output.last_hidden_state[:, 0, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
        
# %%
if __name__ == '__main__':
    import torchinfo
    model = TextClassifier()
    config = model.config
    input_ids = torch.randint(0, config.vocab_size, size=(1, 300))
    torchinfo.summary(model, input_data=input_ids)