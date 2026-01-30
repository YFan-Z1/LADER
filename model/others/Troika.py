import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from model_ori.ThreeBranch import ThreeBranch
from model_ori.common import CrossAttentionLayer


class Troika(ThreeBranch):
    def __init__(self, config, attributes, classes, offset, device, logger):
        super().__init__(config, attributes, classes, offset, device, logger)
        vision_width = self.clip.visual.transformer.width
        self.vision_width = vision_width
        output_dim = self.clip.visual.output_dim
        # cmt
        self.cross_attn_dropout = self.clip_cfg.cross_attn_dropout

        self.cmt = nn.ModuleList([CrossAttentionLayer(output_dim, output_dim // 64, self.cross_attn_dropout) for _ in
                                  range(self.clip_cfg.cmt_layers)])
        self.lamda = nn.Parameter(torch.ones(output_dim) * self.clip_cfg.init_lamda)
        self.patch_norm = nn.LayerNorm(output_dim)

    def loss_calu(self, logits, target, idx):
        loss_fn = CrossEntropyLoss()
        _, batch_attr, batch_obj, batch_pair = target
        pair_logits, attr_logits, obj_logits = logits
        batch_attr = batch_attr.to(self.device)
        batch_obj = batch_obj.to(self.device)
        batch_pair = batch_pair.to(self.device)
        loss_pair = loss_fn(pair_logits, batch_pair) * self.config.pair_loss_weight
        loss_attr = loss_fn(attr_logits, batch_attr) * self.config.attr_loss_weight
        loss_obj = loss_fn(obj_logits, batch_obj) * self.config.obj_loss_weight

        loss = loss_attr + loss_obj + loss_pair
        if self.training:
            return {
                'loss': loss,
                'attr': loss_attr,
                'obj': loss_obj,
                'pair': loss_pair,
            }
        else:
            return loss

    def logit_infer(self, logits, idx):
        pair_logits, attr_logits, obj_logits = logits
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)
        for i_comp in range(pair_logits.shape[-1]):
            a_idx, o_idx = idx[i_comp, 0], idx[i_comp, 1]
            weighted_attr_pred = attr_pred[:, a_idx] * self.config.prim_inference_weight
            weighted_obj_pred = obj_pred[:, o_idx] * self.config.prim_inference_weight
            pair_logits[:, i_comp] = (
                    pair_logits[:, i_comp] * self.config.pair_inference_weight +
                    weighted_attr_pred * weighted_obj_pred
            )
        return pair_logits

    def forward(self, batch, idx):
        batch_img, batch_attr, batch_obj, batch_pair = batch
        batch_img = batch_img.to(self.device)
        b = batch_img.shape[0]
        batch_img = self.encode_image(batch_img.type(self.clip.dtype))
        logits, text_feats, visual_feats = list(), dict(), dict()
        cls_token, patch_token = batch_img[:, 0, ], batch_img[:, 1:, ]
        attr_feat, obj_feat = self.attr_disentangler(cls_token), self.obj_disentangler(cls_token)

        if self.text_prompt_mode == "soft":
            text_feats = self._encode_comp_text_soft(idx)  # dict

        visual_feats['pair'] = cls_token / cls_token.norm(dim=-1, keepdim=True)
        visual_feats['obj'] = obj_feat / obj_feat.norm(dim=-1, keepdim=True)
        visual_feats['attr'] = attr_feat / attr_feat.norm(dim=-1, keepdim=True)

        logits = list()
        for stage in ['pair', 'attr', 'obj']:
            # CMT
            idx_text_features = text_feats[stage]
            cmt_text_features = idx_text_features.unsqueeze(0).expand(b, -1, -1)
            batch_patch = self.patch_norm(patch_token)
            for layer in self.cmt:
                cmt_text_features = layer(cmt_text_features, batch_patch)
            cmt_text_features = idx_text_features + self.lamda * cmt_text_features.squeeze(1)
            cmt_text_features = cmt_text_features / cmt_text_features.norm(dim=-1, keepdim=True)
            logits.append(
                torch.einsum(
                    "bd, bkd->bk",
                    visual_feats[stage],
                    cmt_text_features * self.clip.logit_scale.exp()
                ))
        return logits