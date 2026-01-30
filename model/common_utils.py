import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_conditional_orthogonality(z_a, z_o, z_c):
    # Minimize correlation(z_a, z_o | z_c), First normalize z_c for projection
    z_c_norm = F.normalize(z_c, dim=-1)
    # Project z_a out of z_c, (B, C) * (B, C) -> (B, 1)
    dot_ac = torch.sum(z_a * z_c_norm, dim=-1, keepdim=True)
    z_a_perp = z_a - dot_ac * z_c_norm
    # Project z_o out of z_c
    dot_oc = torch.sum(z_o * z_c_norm, dim=-1, keepdim=True)
    z_o_perp = z_o - dot_oc * z_c_norm
    # Calculate Cosine Similarity of the residuals, Minimize squared cosine similarity
    z_a_perp = F.normalize(z_a_perp, dim=-1)
    z_o_perp = F.normalize(z_o_perp, dim=-1)
    cos_sim = torch.sum(z_a_perp * z_o_perp, dim=-1)
    return torch.mean(cos_sim ** 2)


def compute_base_logits(visual_dict, text_dict, logit_scale):
    logits = list()
    for stage in ['pair', 'attr', 'obj']:
        logits.append(
            torch.einsum("bd, kd->bk", visual_dict[stage],
                         text_dict[stage] * logit_scale.exp())
        )
    return logits


class FiLM(nn.Module):
    def __init__(self, condition_dim, feature_dim):
        super().__init__()
        self.net = nn.Linear(condition_dim, feature_dim * 2)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, z_c):
        params = self.net(z_c)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma, beta


class FiLMBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.film_gen = FiLM(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x, z_c):
        gamma, beta = self.film_gen(z_c)
        modulated = (1 + gamma) * x + beta
        out = self.mlp(self.act(modulated))
        return self.norm(x + out)


class FiLMedDecoder(nn.Module):
    def __init__(self, dim, depth=2, dropout=0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([
            FiLMBlock(dim, dropout) for _ in range(depth)
        ])
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, z_a, z_o, z_c):
        content = self.fusion(torch.cat([z_a, z_o], dim=-1))
        x = content
        for block in self.blocks:
            x = block(x, z_c)
        return self.out_proj(x)