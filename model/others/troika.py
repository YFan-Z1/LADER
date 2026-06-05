import torch
import torch.nn as nn
import torch.nn.functional as F
from model.baseline.ThreeBranch import ThreeBranch
from model.utils.losses import MaxSup
from model.PCI.OTAttention import CrossAttentionLayer
from model.utils.common_utils import expand_tensor, custom_visual_feat

class Troika(ThreeBranch):
    def __init__(self, config, attributes, classes, offset, device, logger):
        super().__init__(config, attributes, classes, offset, device, logger)
        output_dim = self.clip.visual.output_dim
        self.patch_norm = nn.LayerNorm(output_dim)
        self.lamda = nn.Parameter(torch.ones(output_dim) * 0.1)
        self.cmt = nn.ModuleList([CrossAttentionLayer(output_dim, output_dim // 64, self.clip_cfg.dropout, mlp_ratio=4.0) for _ in
                                  range(self.clip_cfg.cmt_layers)])
        # criterion
        use_maxsup = self.config.maxsup_smooth
        self.criterion = nn.CrossEntropyLoss() if not use_maxsup else MaxSup(self.config.smoothing)

    def loss_calu(self, logits, target, idx):
        loss_fn = self.criterion
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
        pair_logits, attr_logits, obj_logits = logits[:3]
        idx = idx.to(device=pair_logits.device, dtype=torch.long)
        a_idx, o_idx = idx[:, 0], idx[:, 1]
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)
        attr_score = attr_pred.index_select(dim=1, index=a_idx) * self.config.prim_inference_weight
        obj_score = obj_pred.index_select(dim=1, index=o_idx) * self.config.prim_inference_weight
        pair_logits = pair_logits * self.config.pair_inference_weight + attr_score * obj_score
        return pair_logits

    def forward_for_open(self, batch, text_feats):
        ### extract base visual feature for attr\obj\comp
        batch_img = batch[0]
        batch_img, b = batch_img.to(self.device), batch_img.shape[0]
        batch_img = self.encode_image(batch_img.type(self.clip.dtype))
        cls_token, patch_token = batch_img[:, 0, :], batch_img[:, 1:, :]
        attr_feat, obj_feat = self.attr_disentangler(cls_token), self.obj_disentangler(cls_token)
        visual_feats = custom_visual_feat(cls_token, attr_feat, obj_feat)

        ### extract base text feature for attr\obj\comp
        text_attr_feats = expand_tensor(text_feats['attr'], b)
        text_obj_feats = expand_tensor(text_feats['obj'], b)
        text_pair_feats = expand_tensor(text_feats['pair'], b)
        temp_text_feats = custom_visual_feat(text_pair_feats, text_attr_feats, text_obj_feats)

        logits = list()
        for stage in ['pair', 'attr', 'obj']:
            cmt_text_features = temp_text_feats[stage]
            batch_patch = self.patch_norm(patch_token)
            for layer in self.cmt:
                cmt_text_features = layer(cmt_text_features, batch_patch)
            cmt_text_features = temp_text_feats[stage] + self.lamda * cmt_text_features.squeeze(1)
            cmt_text_features = cmt_text_features / cmt_text_features.norm(dim=-1, keepdim=True)
            logit = torch.einsum("bd, bkd->bk", visual_feats[stage], cmt_text_features * self.clip.logit_scale.exp())
            logits.append(logit)
        del temp_text_feats
        return logits

    def forward(self, batch, idx):
        ### extract base visual feature for attr\obj\comp
        batch_img = batch[0]
        batch_img, b = batch_img.to(self.device), batch_img.shape[0]
        batch_img = self.encode_image(batch_img.type(self.clip.dtype))
        cls_token, patch_token = batch_img[:, 0, :], batch_img[:, 1:, :]
        attr_feat, obj_feat = self.attr_disentangler(cls_token), self.obj_disentangler(cls_token)
        visual_feats = custom_visual_feat(cls_token, attr_feat, obj_feat)

        ### extract base text feature for attr\obj\comp

        text_feats = self._encode_comp_text_soft(idx)
        text_feats['attr'] = expand_tensor(text_feats['attr'], b)
        text_feats['obj'] = expand_tensor(text_feats['obj'], b)
        text_feats['pair'] = expand_tensor(text_feats['pair'], b)
        logits = list()
        for stage in ['pair', 'attr', 'obj']:
            cmt_text_features = text_feats[stage]
            batch_patch = self.patch_norm(patch_token)
            for layer in self.cmt:
                cmt_text_features = layer(cmt_text_features, batch_patch)
            cmt_text_features = text_feats[stage] + self.lamda * cmt_text_features.squeeze(1)
            cmt_text_features = cmt_text_features / cmt_text_features.norm(dim=-1, keepdim=True)
            logit = torch.einsum("bd, bkd->bk", visual_feats[stage], cmt_text_features * self.clip.logit_scale.exp())
            logits.append(logit)
        return logits
