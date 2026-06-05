import torch.nn.functional as F
import torch.nn as nn
import torch
from model.thirdparty.rms_norm import RMSNorm
from model.thirdparty.layers import MLP, SwiGLUFFN

class Disentangler(nn.Module):
    def __init__(self, emb_dim):
        super(Disentangler, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x
    
class AttnPoolHead(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=8, mlp_type='SwiGLUFFN', mlp_ratio=2.0):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.attention = torch.nn.MultiheadAttention(hidden_size, num_attention_heads, batch_first=True)
        self.norm = RMSNorm(hidden_size)
        if mlp_type not in ['mlp', 'MLP', 'SwiGLUFFN']:
            raise NotImplementedError
        self.mlp = MLP(hidden_size, int(hidden_size * mlp_ratio), hidden_size) if mlp_type in ['mlp', "MLP"] else SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), hidden_size)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.norm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]