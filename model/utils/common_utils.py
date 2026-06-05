import torch
import torch.nn as nn
import torch.nn.functional as F

def custom_visual_feat(comp_feat, attr_feat, obj_feat):
    visual_feats = dict()
    visual_feats['pair'] = comp_feat / comp_feat.norm(dim=-1, keepdim=True)
    visual_feats['obj'] = obj_feat / obj_feat.norm(dim=-1, keepdim=True)
    visual_feats['attr'] = attr_feat / attr_feat.norm(dim=-1, keepdim=True)
    return visual_feats

def compute_base_logits(visual_dict, text_dict, logit_scale):
    logits = list()
    for stage in ['pair', 'attr', 'obj']:
        logit = torch.einsum("bd, kd->bk", visual_dict[stage], text_dict[stage] * logit_scale.exp())
        # if stage == text_dict[stage].dim() == 2:
        #     logit = torch.einsum("bd, kd->bk", visual_dict[stage], text_dict[stage] * logit_scale.exp())
        # else:
        #     logit = torch.einsum("bd, bkd->bk", visual_dict[stage], text_dict[stage] * logit_scale.exp())
        logits.append(logit)
    return logits

def expand_tensor(x: torch.Tensor, b: int) -> torch.Tensor:
    x = x.unsqueeze(0).expand(b, -1, -1)
    return x