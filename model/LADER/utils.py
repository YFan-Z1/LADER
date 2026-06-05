import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.thirdparty.layers import SwiGLUFFN
from model.thirdparty.rms_norm import RMSNorm


class ConfigAccessor:
    def __init__(self, *configs):
        self.configs = configs

    def value(self, names, default):
        for cfg in self.configs:
            for name in names:
                if hasattr(cfg, name):
                    return getattr(cfg, name)
        return default

    def int(self, names, default):
        value = int(self.value(names, default))
        if value < 1:
            raise ValueError(f"{names[0]} must be >= 1, got {value}")
        return value

    def nonnegative_int(self, names, default):
        value = int(self.value(names, default))
        if value < 0:
            raise ValueError(f"{names[0]} must be >= 0, got {value}")
        return value

    def bool(self, names, default):
        value = self.value(names, default)
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def optional_float(self, names, default):
        value = self.value(names, default)
        if value is None:
            return None
        return float(value)


def build_active_mask(
    num_base_tokens,
    query_sizes,
    device,
    dtype=None,
    allow_primitive_cls_attention=False,
):
    total_queries = sum(query_sizes)
    total_len = num_base_tokens + total_queries
    mask_kwargs = {"device": device}
    if dtype is not None:
        mask_kwargs["dtype"] = dtype
    mask = torch.zeros(total_len, total_len, **mask_kwargs)

    if total_queries == 0:
        return mask

    q_start = num_base_tokens
    cls_idx = 0
    patch_slice = slice(1, q_start)
    attr_slice, obj_slice, ctx_slice = query_slices(query_sizes, offset=q_start)

    mask[:q_start, q_start:] = float("-inf")

    mask[attr_slice, cls_idx] = 0.0 if allow_primitive_cls_attention else float("-inf")
    mask[obj_slice, cls_idx] = 0.0 if allow_primitive_cls_attention else float("-inf")
    mask[attr_slice, patch_slice] = 0.0
    mask[obj_slice, patch_slice] = 0.0
    mask[attr_slice, obj_slice] = float("-inf")
    mask[obj_slice, attr_slice] = float("-inf")
    mask[attr_slice, ctx_slice] = float("-inf")
    mask[obj_slice, ctx_slice] = float("-inf")

    mask[ctx_slice, attr_slice] = float("-inf")
    mask[ctx_slice, obj_slice] = float("-inf")
    return mask


def query_slices(query_sizes, offset=0):
    ends, start = [], offset
    for size in query_sizes:
        ends.append(slice(start, start + size))
        start += size
    return tuple(ends)


def split_query_groups(query_tokens, query_sizes):
    return torch.split(query_tokens, query_sizes, dim=1)


def pool_query_groups(query_tokens, query_sizes, mode="similarity", groups=None):
    groups = groups if groups is not None else split_query_groups(query_tokens, query_sizes)
    if mode == "similarity":
        attr_group, obj_group, ctx_group = groups
        return (
            similarity_pool(attr_group),
            similarity_pool(obj_group),
            similarity_pool(ctx_group),
        )
    if mode == "first":
        return tuple(group[:, 0] for group in groups)
    if mode == "max":
        return tuple(group.max(dim=1).values for group in groups)
    if mode != "mean":
        raise ValueError(f"Unknown query_pooling mode: {mode}")
    return tuple(group.mean(dim=1) for group in groups)


def similarity_pool(group, eps=1e-6):
    if group.size(1) == 1:
        return group[:, 0]

    normalized = F.normalize(group, dim=-1, eps=eps)
    affinity = torch.matmul(normalized, normalized.transpose(-1, -2))
    scores = affinity.mean(dim=-1)
    weights = scores.softmax(dim=-1)
    return torch.sum(group * weights.unsqueeze(-1), dim=1)


def _softplus_inverse(x):
    x = float(max(x, 1e-6))
    return x + math.log(-math.expm1(-x))


def normalized_entropy_from_logits(logits, dim=-1, eps=1e-8):
    probs = F.softmax(logits, dim=dim)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=dim)
    return entropy / math.log(max(logits.size(dim), 2))


def primitive_uncertainty(attr_logits, obj_logits):
    return 0.5 * (
        normalized_entropy_from_logits(attr_logits) +
        normalized_entropy_from_logits(obj_logits)
    )


class FiLM(nn.Module):
    def __init__(self, condition_dim, feature_dim):
        super().__init__()
        self.net = nn.Linear(condition_dim, feature_dim * 2)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, condition):
        gamma, beta = self.net(condition).chunk(2, dim=-1)
        return gamma, beta


class FiLMBlock(nn.Module):
    def __init__(self, dim, condition_dim=None, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        condition_dim = condition_dim or dim
        hidden_dim = int(dim * mlp_ratio)
        self.norm = RMSNorm(dim)
        self.film = FiLM(condition_dim, dim)
        self.ffn = SwiGLUFFN(dim, hidden_dim, dim, drop=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, condition):
        gamma, beta = self.film(condition)
        x_norm = self.norm(x)
        x_mod = x_norm * (1.0 + gamma) + beta
        return x + self.drop(self.ffn(x_mod))


class FiLMedDecoder(nn.Module):
    def __init__(self, dim, hidden_dim=None, depth=2, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.in_proj = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            RMSNorm(hidden_dim),
            nn.GELU(),
        )
        self.ctx_proj = nn.Linear(dim, hidden_dim) if hidden_dim != dim else nn.Identity()
        self.blocks = nn.ModuleList([
            FiLMBlock(hidden_dim, hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.out_norm = RMSNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, z_a, z_o, z_c):
        x = self.in_proj(torch.cat([z_a, z_o], dim=-1))
        ctx = self.ctx_proj(z_c)
        for block in self.blocks:
            x = block(x, ctx)
        return self.out_proj(self.out_norm(x))


class AdaLNZeroSwiGLUBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        init_res_scale: float = 1e-3,
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.norm = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, hidden_dim, dim, drop=dropout)
        self.drop = nn.Dropout(dropout)

        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 3))

        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

        self.init_res_scale = init_res_scale

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta, alpha = self.modulation(cond).chunk(3, dim=-1)
        h = self.norm(x)
        h = h * (1.0 + gamma) + beta
        h = self.ffn(h)
        gate = self.init_res_scale + torch.tanh(alpha)
        return x + self.drop(h) * gate


class EvidenceTransportBinder(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        temperature: float = 0.07,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.a_score = nn.Linear(dim, dim, bias=False)
        self.o_score = nn.Linear(dim, dim, bias=False)
        self.ctx_to_a = nn.Linear(dim, dim, bias=False)
        self.ctx_to_o = nn.Linear(dim, dim, bias=False)
        self.a_pair = nn.Linear(dim, hidden_dim, bias=False)
        self.o_pair = nn.Linear(dim, hidden_dim, bias=False)
        self.i_pair = nn.Linear(dim, hidden_dim, bias=True)
        self.out = nn.Sequential(
            RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q_a: torch.Tensor,
        q_o: torch.Tensor,
        z_c: torch.Tensor,
    ) -> torch.Tensor:
        a_ctx = self.ctx_to_a(z_c).unsqueeze(1)
        o_ctx = self.ctx_to_o(z_c).unsqueeze(1)

        qa_score = F.normalize(self.a_score(q_a) + a_ctx, dim=-1)
        qo_score = F.normalize(self.o_score(q_o) + o_ctx, dim=-1)

        scores = torch.matmul(qa_score, qo_score.transpose(1, 2))
        scores = scores / max(self.temperature, 1e-6)

        weights = scores.flatten(1).softmax(dim=-1)
        weights = weights.view_as(scores)

        a_weight = weights.sum(dim=2)
        o_weight = weights.sum(dim=1)
        a_summary = torch.sum(q_a * a_weight.unsqueeze(-1), dim=1)
        o_summary = torch.sum(q_o * o_weight.unsqueeze(-1), dim=1)
        interaction = torch.einsum("bmn,bmd,bnd->bd", weights, q_a, q_o)
        evidence = self.a_pair(a_summary) + self.o_pair(o_summary) + self.i_pair(interaction)

        return self.out(evidence)


class PrimitiveBaseComposer(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.a_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.i_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.norm = RMSNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, z_a: torch.Tensor, z_o: torch.Tensor) -> torch.Tensor:
        x = self.a_proj(z_a) + self.o_proj(z_o) + self.i_proj(z_a * z_o)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class ContextBoundedManifoldDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        depth: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        use_transport: bool = True,
        transport_temperature: float = 0.07,
        max_residual_scale: float = 0.35,
        normalize_output: bool = False,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim

        self.dim, self.hidden_dim = dim, hidden_dim
        self.use_transport = use_transport
        self.max_residual_scale = max_residual_scale
        self.normalize_output = normalize_output

        self.base_composer = PrimitiveBaseComposer(
            dim=dim, hidden_dim=hidden_dim, dropout=dropout,
        )

        if use_transport:
            self.evidence_binder = EvidenceTransportBinder(
                dim=dim,
                hidden_dim=hidden_dim,
                temperature=transport_temperature,
                dropout=dropout,
            )
            self.evidence_gate = nn.Parameter(torch.tensor(0.0))
        else:
            self.evidence_binder = None
            self.evidence_gate = None

        self.ctx_proj = nn.Sequential(nn.Linear(dim, hidden_dim),
            RMSNorm(hidden_dim), nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            AdaLNZeroSwiGLUBlock(
                dim=hidden_dim,
                cond_dim=hidden_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.anchor_norm = RMSNorm(hidden_dim)
        self.anchor_proj = nn.Linear(hidden_dim, dim)

        self.res_norm = RMSNorm(hidden_dim)
        self.res_proj = nn.Linear(hidden_dim, dim)

        self.ctx_scale = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.ctx_scale.weight)
        nn.init.constant_(self.ctx_scale.bias, -4.0)

    @staticmethod
    def _project_to_tangent(
        residual: torch.Tensor,
        anchor: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        anchor_unit = F.normalize(anchor, dim=-1, eps=eps)
        parallel = (residual * anchor_unit).sum(dim=-1, keepdim=True) * anchor_unit
        return residual - parallel

    def forward(
        self,
        z_a: torch.Tensor,
        z_o: torch.Tensor,
        z_c: torch.Tensor,
        q_a: torch.Tensor = None,
        q_o: torch.Tensor = None,
    ) -> torch.Tensor:
        h = self.base_composer(z_a, z_o)

        if self.use_transport and q_a is not None and q_o is not None:
            evidence = self.evidence_binder(q_a, q_o, z_c)
            h = h + torch.tanh(self.evidence_gate) * evidence
        anchor = self.anchor_proj(self.anchor_norm(h))
        ctx = self.ctx_proj(z_c)
        x = h
        for block in self.blocks:
            x = block(x, ctx)
        residual = self.res_proj(self.res_norm(x))
        residual = self._project_to_tangent(residual, anchor)
        scale = self.max_residual_scale * torch.sigmoid(self.ctx_scale(ctx))
        z_rec = anchor + scale * residual
        if self.normalize_output:
            z_rec = F.normalize(z_rec, dim=-1)
        return z_rec
