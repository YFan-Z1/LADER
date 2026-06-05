import torch.nn.functional as F
import torch.nn as nn
import torch

class MaxSup(nn.Module):
    def __init__(self, smooth_weight: int):
        super(MaxSup, self).__init__()
        self.base_criterion = nn.CrossEntropyLoss()
        self.smooth_weight = smooth_weight

    def forward(self, logit, label):
        base_loss = self.base_criterion(logit, label)
        z_max = torch.max(logit, dim=1, keepdim=True)[0]
        z_mean = torch.mean(logit, dim=1, keepdim=True)
        aux_loss = self.smooth_weight * (z_max - z_mean)
        return base_loss + aux_loss.mean()

def compute_ot_loss(vanilla, pi_star, epsilon=1e-8):
    ot_loss = 0.0
    for v, s in zip(vanilla, pi_star):
        v = v.clamp_min(epsilon)
        s = s.clamp_min(epsilon)
        v = v / v.sum(dim=-1, keepdim=True)
        s = s / s.sum(dim=-1, keepdim=True)
        M = 0.5 * (v + s)
        kl1 = 0.5 * F.kl_div((M + epsilon).log(), v + epsilon, reduction='sum') / v.size(0)
        kl2 = 0.5 * F.kl_div((M + epsilon).log(), s + epsilon, reduction='sum') / v.size(0)
        js_div = kl1 + kl2
        ot_loss += js_div
    return ot_loss

def compute_ortho_loss(proto):
    proto = F.normalize(proto, dim=-1)
    sim = torch.mm(proto, proto.t())
    mask = ~torch.eye(sim.size(1), device=sim.device, dtype=torch.bool)
    off_diag = sim.masked_select(mask).abs()
    return off_diag.mean()

def compute_align_loss(proto, text, temperature=1.0):
    proto = F.normalize(proto, dim=-1)
    text = F.normalize(text.detach(), dim=-1)
    score_p = torch.mm(proto, text.t()) / temperature
    score_q = score_p.t()
    p = F.softmax(score_p, dim=-1)
    q = F.softmax(score_q, dim=-1)
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(m.log(), p, reduction='batchmean')
    kl_qm = F.kl_div(m.log(), q, reduction='batchmean')
    return 0.5 * (kl_pm + kl_qm)


def compute_conditional_orthogonality(z_attr, z_obj, z_ctx, eps=1e-6):
    z_ctx = F.normalize(z_ctx, dim=-1, eps=eps)
    attr_res = z_attr - (z_attr * z_ctx).sum(dim=-1, keepdim=True) * z_ctx
    obj_res = z_obj - (z_obj * z_ctx).sum(dim=-1, keepdim=True) * z_ctx
    attr_res = F.normalize(attr_res, dim=-1, eps=eps)
    obj_res = F.normalize(obj_res, dim=-1, eps=eps)
    return (attr_res * obj_res).sum(dim=-1).pow(2).mean()
