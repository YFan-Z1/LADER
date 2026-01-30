import torch
import torch.nn as nn
import torch.nn.functional as F
from model_ori.common import CrossAttentionLayer
from model_ori.ThreeBranch import ThreeBranch
from model_ori.losses import compute_conditional_orthogonality, MaxSup
from model_ori.common_utils import FiLMedDecoder, compute_base_logits


class TroikaLADER(ThreeBranch):
    def __init__(self, config, attributes, classes, offset, device, logger):
        super().__init__(config, attributes, classes, offset, device, logger)
        del self.attr_disentangler, self.obj_disentangler
        vision_width = self.clip.visual.transformer.width
        self.vision_width = vision_width
        output_dim = self.clip.visual.output_dim
        self.q_attr = nn.Parameter(torch.randn(1, 1, vision_width) * 0.02)
        self.q_obj = nn.Parameter(torch.randn(1, 1, vision_width) * 0.02)
        self.q_ctx = nn.Parameter(torch.randn(1, 1, vision_width) * 0.02)
        self.rectifier = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
            nn.Dropout(self.clip_cfg.rectifier_dropout),
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )
        self.filmed_decoder = FiLMedDecoder(dim=output_dim, depth=self.clip_cfg.decoder_depth,
                                            dropout=self.clip_cfg.decoder_dropout)

        # criterion
        use_maxsup = self.config.maxsup_smooth
        self.criterion = nn.CrossEntropyLoss() if not use_maxsup else MaxSup(self.config.smoothing)

        # cmt
        self.cross_attn_dropout = self.clip_cfg.cross_attn_dropout

        self.cmt = nn.ModuleList([CrossAttentionLayer(output_dim, output_dim // 64, self.cross_attn_dropout) for _ in
                                  range(self.clip_cfg.cmt_layers)])
        self.lamda = nn.Parameter(torch.ones(output_dim) * self.clip_cfg.init_lamda)
        self.patch_norm = nn.LayerNorm(output_dim)

    def build_active_mask(self, L, device):
        total_len = L + 3
        mask = torch.zeros((total_len, total_len), device=device)
        mask[:L, L:] = float('-inf')
        mask[L, L + 1] = float('-inf')
        mask[L + 1, L] = float('-inf')
        return mask

    def encode_image_with_adapter(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # [B, C, H, W]
        B, C, H_grid, W_grid = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, HW]
        x = x.permute(0, 2, 1)  # [B, HW, C]
        x = torch.cat(
            [self.clip.visual.class_embedding.to(x.dtype)
             + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1, )  # [B, HW+1, C]
        pos_embed = self.clip.visual.positional_embedding
        if x.shape[1] != pos_embed.shape[0]:
            pos_embed = self.resize_pos_embed(pos_embed, H_grid, W_grid)
        x = x + pos_embed.to(x.dtype)
        x = self.clip.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # [L, B, C] (Transformer 期望 seq_len 在第一维)
        L, _, _ = x.shape
        inject_layer_idx = self.clip_cfg.inject_layer_idx
        q_a = self.q_attr.expand(-1, B, -1).to(x.dtype)
        q_o = self.q_obj.expand(-1, B, -1).to(x.dtype)
        q_c = self.q_ctx.expand(-1, B, -1).to(x.dtype)
        active_mask = self.build_active_mask(L, x.device).to(x.dtype)
        current_seq = x
        for i_block in range(self.clip.visual.transformer.layers):
            block = self.clip.visual.transformer.resblocks[i_block]
            if i_block == inject_layer_idx:
                current_seq = torch.cat([current_seq, q_a, q_o, q_c], dim=0)
            attn_mask = active_mask if i_block >= inject_layer_idx else None
            # MHA
            adapt_x = self.peft_tuner[i_block](current_seq, add_residual=False)
            norm_x = block.ln_1(current_seq)
            attn_output, _ = block.attn(
                query=norm_x, key=norm_x, value=norm_x,
                need_weights=False,
                attn_mask=attn_mask
            )
            current_seq = current_seq + attn_output + adapt_x
            # MLP
            residual = current_seq
            norm_x = block.ln_2(current_seq)
            mlp_output = block.mlp(norm_x)
            i_adapter = i_block + self.clip.visual.transformer.layers
            adapt_x = self.peft_tuner[i_adapter](current_seq, add_residual=False)
            current_seq = residual + mlp_output + adapt_x
        final_seq = current_seq.permute(1, 0, 2)
        img_feature = final_seq[:, :L, :]
        img_feature = self.clip.visual.ln_post(img_feature)
        q_features = self.clip.visual.ln_post(final_seq[:, L:, :])  # [B, 3, C]
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
            q_features = q_features @ self.clip.visual.proj
        z_a = q_features[:, 0, :]  # Attribute
        z_o = q_features[:, 1, :]  # Object
        z_c = q_features[:, 2, :]  # Context
        return img_feature, z_a, z_o, z_c

    def loss_calu(self, logits, target, idx):
        loss_fn = self.criterion
        _, batch_attr, batch_obj, batch_pair = target
        pair_logits, attr_logits, obj_logits, comp_logits, loss_co = logits
        # pair_logits, attr_logits, obj_logits, loss_co = logits
        batch_attr = batch_attr.to(self.device)
        batch_obj = batch_obj.to(self.device)
        batch_pair = batch_pair.to(self.device)
        loss_comp = loss_fn(comp_logits, batch_pair) * self.config.comp_loss_weight
        loss_pair = loss_fn(pair_logits, batch_pair) * self.config.pair_loss_weight
        loss_attr = loss_fn(attr_logits, batch_attr) * self.config.attr_loss_weight
        loss_obj = loss_fn(obj_logits, batch_obj) * self.config.obj_loss_weight

        loss = loss_comp + loss_attr + loss_obj + loss_pair + loss_co
        # loss = loss_attr + loss_obj + loss_pair + loss_co
        if self.training:
            return {
                'loss': loss,
                'loss_c': loss_comp,
                'loss_a': loss_attr,
                'loss_o': loss_obj,
                'loss_p': loss_pair,
                'loss_co': loss_co
            }
        else:
            return loss

    def logit_infer(self, logits, idx):
        pair_logits, attr_logits, obj_logits, comp_logits, _ = logits
        # pair_logits, attr_logits, obj_logits, _ = logits
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)
        for i_comp in range(pair_logits.shape[-1]):
            a_idx, o_idx = idx[i_comp, 0], idx[i_comp, 1]
            weighted_attr_pred = attr_pred[:, a_idx] * self.config.prim_inference_weight
            weighted_obj_pred = obj_pred[:, o_idx] * self.config.prim_inference_weight
            pair_logits[:, i_comp] = (
                    pair_logits[:, i_comp] * self.config.pair_inference_weight +
                    comp_logits[:, i_comp] * self.config.comp_inference_weight +
                    weighted_attr_pred * weighted_obj_pred
            )
        return pair_logits

    def forward_for_open(self, batch, text_feature):
        batch_img = batch[0]
        batch_img = batch_img.to(self.device)
        img_feature, z_a, z_o, z_c = self.encode_image_with_adapter(batch_img.type(self.clip.dtype))

        logits, text_feats, visual_feats = list(), dict(), dict()
        cls_token = img_feature[:, 0, :]
        gate = self.rectifier(torch.cat([z_a, z_c], dim=-1))
        z_a = z_a * (1 + gate)
        visual_feats['pair'] = cls_token / cls_token.norm(dim=-1, keepdim=True)
        visual_feats['obj'] = z_o / z_o.norm(dim=-1, keepdim=True)
        visual_feats['attr'] = z_a / z_a.norm(dim=-1, keepdim=True)
        # reverse the text feats
        reverse_idx_mapping = {v: k for k, v in self.idx_mapping.items()}
        for idx in range(len(text_feature)):
            stage = reverse_idx_mapping[idx]
            text_feats[stage] = text_feature[idx]
        del text_feature
        logits = compute_base_logits(visual_feats, text_feats, self.clip.logit_scale)

        # decode compositional feature
        comp_feats = self.filmed_decoder(z_a, z_o, z_c)
        # comp_feats = self.filmed_decoder(z_a.detach(), z_o.detach(), z_c.detach())
        comp_feats = comp_feats / comp_feats.norm(dim=-1, keepdim=True)
        comp_logits = torch.einsum("bd, kd->bk", comp_feats, text_feats['pair'] * self.clip.logit_scale.exp())
        logits.append(comp_logits)

        loss_co = compute_conditional_orthogonality(z_a, z_o, z_c) * self.config.cond_orth_loss_weight
        logits.append(loss_co)

        return logits

    def forward(self, batch, idx):
        batch_img = batch[0]
        batch_img = batch_img.to(self.device)
        B = batch_img.shape[0]
        img_feature, z_a, z_o, z_c = self.encode_image_with_adapter(batch_img.type(self.clip.dtype))

        logits, text_feats, visual_feats = list(), dict(), dict()
        cls_token, patch_token = img_feature[:, 0, :], img_feature[:, 1:, :]
        gate = self.rectifier(torch.cat([z_a, z_c], dim=-1))
        z_a = z_a * (1 + gate)
        visual_feats['pair'] = cls_token / cls_token.norm(dim=-1, keepdim=True)
        visual_feats['obj'] = z_o / z_o.norm(dim=-1, keepdim=True)
        visual_feats['attr'] = z_a / z_a.norm(dim=-1, keepdim=True)
        text_feats, logits = self._encode_comp_text_soft(idx), list()
        # cmt module
        for stage in ['pair', 'attr', 'obj']:
            idx_text_features = text_feats[stage]
            cmt_text_features = idx_text_features.unsqueeze(0).expand(B, -1, -1)
            batch_patch = self.patch_norm(patch_token)
            for layer in self.cmt:
                cmt_text_features = layer(cmt_text_features, batch_patch)
            cmt_text_features = idx_text_features + self.lamda * cmt_text_features.squeeze(1)
            cmt_text_features = cmt_text_features / cmt_text_features.norm(dim=-1, keepdim=True)
            logits.append(torch.einsum("bd, bkd->bk", visual_feats[stage], cmt_text_features * self.clip.logit_scale.exp()))

        # decode compositional feature
        comp_feats = self.filmed_decoder(z_a, z_o, z_c)
        comp_feats = comp_feats / comp_feats.norm(dim=-1, keepdim=True)
        comp_logits = torch.einsum("bd, kd->bk", comp_feats, text_feats['pair'] * self.clip.logit_scale.exp())
        logits.append(comp_logits)

        loss_co = compute_conditional_orthogonality(z_a, z_o, z_c) * self.config.cond_orth_loss_weight
        logits.append(loss_co)

        return logits