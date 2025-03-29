import torch
import torch.nn as nn
import math

class SHSAttention(nn.Module):
    def __init__(self, config, vis):
        super(SHSAttention, self).__init__()
        self.vis = vis
        self.hidden_size = config.hidden_size
        
        # Calculate partial dimension (r = 1/4.67 from SHViT paper)
        self.pdim = int(self.hidden_size / 4.67) #as pare the SHViT paper default 
        self.qk_dim = 16  # Fixed per SHViT design
        
        # Normalization for partial channels (LayerNorm for 1D sequence)
        self.pre_norm = nn.LayerNorm(self.pdim)
        
        # Combined QKV projection for partial channels
        self.qkv = nn.Linear(self.pdim, 2*self.qk_dim + self.pdim)
        
        self.scale = self.qk_dim ** -0.5  # Scaling factor
        
        # Final projection (maintains dimension)
        self.out = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_states):
        B, seq_len, _ = hidden_states.shape
        
        # Split into attended (pdim) and residual channels
        x1 = hidden_states[..., :self.pdim]  # (B, seq_len, pdim)
        x2 = hidden_states[..., self.pdim:]   # (B, seq_len, hidden_size-pdim)
        
        # Normalize partial channels
        x1 = self.pre_norm(x1)  # (B, seq_len, pdim)
        
        # Generate Q, K, V from partial channels
        qkv = self.qkv(x1)  # (B, seq_len, 2*qk_dim + pdim)
        q, k, v = torch.split(qkv, [self.qk_dim, self.qk_dim, self.pdim], dim=-1)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, seq_len, seq_len)
        attn_probs = self.softmax(attn_scores)
        weights = attn_probs if self.vis else None
        attn_probs = self.attn_dropout(attn_probs)
        
        # Attend to values
        attended_x1 = torch.matmul(attn_probs, v)  # (B, seq_len, pdim)
        
        # Concatenate with residual channels
        output = torch.cat([attended_x1, x2], dim=-1)  # (B, seq_len, hidden_size)
        
        # Final projection
        output = self.out(output)
        output = self.proj_dropout(output)
        
        return output, weights