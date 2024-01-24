import torch.nn as nn

class LSTMTextClsModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, num_classes):
        super(LSTMTextClsModel, self).__init_()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=emb_dim)
        
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, 
                            batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        output, _ =  self.rnn(x)
        x = output[:, -1, :]
        x = self.fc(x)
        
        return x