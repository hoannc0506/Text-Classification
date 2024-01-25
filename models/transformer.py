import torch.nn as nn
import torch

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0, bias=True)
        self.ffn = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=ff_dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=ff_dim, out_features=embed_dim, bias=True)
        )
        self.layernorm_1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm_2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
       
        attn_output = self.dropout_1(attn_output)
        out_1 = self.layernorm_1(query + attn_output)
        ffn_output = self.ffn(out_1)
        ffn_output = self.dropout_2(ffn_output)
        out_2 = self.layernorm_2(out_1 + ffn_output)
        return out_2


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length, device):
        super().__init__()
        self.device = device
        self.word_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.pos_emb = nn.Embedding(num_embeddings=max_length, embedding_dim=embed_dim)
        
    def forward(self, x):
        N, seq_len = x.size()
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        outputs1 = self.word_emb(x)
        outputs2 = self.pos_emb(positions)
        
        return outputs1 + outputs2
    
class TransformerTextCls(nn.Module):
    def __init__(self, vocab_size, max_length, num_classes, embed_dim, num_heads, ff_dim, dropout, device):
        super().__init__()
        self.embed_layer = TokenAndPositionEmbedding(vocab_size, embed_dim, max_length, device)
        self.tranformer_layer = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        self.pooling = nn.AvgPool1d(kernel_size=max_length)
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.embed_layer(x)
        x = self.tranformer_layer(query=x, key=x, value=x)
    
        x = self.pooling(x.permute(0,2,1)).squeeze()
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
if __name__ == "__main__":
    import torchinfo
    
    model = TransformerTextCls(vocab_size=1000, 
                               max_length=20, 
                               num_classes=2, 
                               embed_dim=32, 
                               num_heads=2, 
                               ff_dim=32, 
                               dropout=0.1, 
                               device='cuda')
    
    torchinfo.summary(model, input_size=[(30, 8)], dtypes=[torch.long])
    
        
