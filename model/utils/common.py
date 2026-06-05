import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomTextEncoder(torch.nn.Module):
    def __init__(self, clip_model, tokenizer, config, dtype=torch.float16, logger=None):
        super().__init__()
        self.dtype = dtype
        self.config = config
        self.tokenizer = tokenizer
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding.to(dtype)
        self.ln_final = clip_model.ln_final.to(dtype)
        self.text_projection = clip_model.text_projection.to(dtype)
        self.token_embedding = clip_model.token_embedding.to(dtype)

    def tokenize(self, text):
        return torch.cat([self.tokenizer(tok) for tok in text])

    def encode_text(self, text, enable_pos_emb=True):
        token_ids = self.tokenize(text)
        text_features, _ = self.forward(token_ids, None, enable_pos_emb)
        return text_features

    def forward(self, token_ids, token_tensors=None, enable_pos_emb=True):
        if token_tensors is not None:
            text_features = token_tensors.to(self.dtype)
        else:
            text_features = self.token_embedding(token_ids).to(self.dtype)

        text_features = text_features.type(self.dtype)
        x = (
            text_features + self.positional_embedding.type(self.dtype)
            if enable_pos_emb
            else text_features
        )
        x = x.permute(1, 0, 2)
        text_feature = self.transformer(x)
        x = text_feature.permute(1, 0, 2)
        x_post_ln = self.ln_final(x)
        tf = (x_post_ln[torch.arange(x_post_ln.shape[0]), token_ids.argmax(dim=-1)] @ self.text_projection)
        return tf, text_feature


class Disentangler(nn.Module):
    def __init__(self, emb_dim):
        super(Disentangler, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x


class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
