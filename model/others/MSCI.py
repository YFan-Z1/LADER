from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
from model.common import *
from model_ori.ThreeBranch import ThreeBranch
from model_ori.common import CrossAttentionLayer


class FeatureVariance(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, features, text_features):
        features_flat = features.view(-1, self.feature_dim)
        # Calculate dot product similarity
        similarity_scores = torch.matmul(features_flat, text_features.t())  # [batch_size * seq_len, num_text]
        # Calculate variance of similarity scores along text feature dimension
        variance = torch.var(similarity_scores, dim=1, unbiased=False)  # [batch_size * seq_len]
        # Reshape back to [batch_size, seq_len]
        return variance.view(features.size(0), features.size(1))


class LocalFeatureAdd(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.variance_calc = FeatureVariance(feature_dim)

    def forward(self, low_features, high_features, text_features):
        low_variances = self.variance_calc(low_features, text_features)
        global_low_variance = low_variances[:, 0].unsqueeze(1)
        low_variances_ratio = low_variances / global_low_variance
        high_variances = self.variance_calc(high_features, text_features)
        global_high_variance = high_variances[:, 0].unsqueeze(1)
        high_variances_ratio = high_variances / global_high_variance
        ratio = low_variances_ratio / high_variances_ratio
        enhance_ratio = torch.clamp(ratio, min=1)
        # print(enhance_ratio.shape)
        enhanced_high_feature = high_features * enhance_ratio.unsqueeze(2)
        return enhanced_high_feature


class Stage1LocalAlignment(nn.Module):
    def __init__(self, d_model, num_heads=12, dropout=0.1, num_cmt_layers_first=1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # self.spatial_attn = SpatialAttention(d_model)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout) for _ in range(num_cmt_layers_first)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv_low_level):
        kv_low_level = self.norm(kv_low_level)
        # kv_low_level=self.se_attn_layers(kv_low_level)
        for cross_attn in self.cross_attn_layers:
            q = cross_attn(q, kv_low_level)
        return q


class Stage2GlobalFusion(nn.Module):
    def __init__(self, d_model, num_heads=12, dropout=0.1, num_cmt_layers_second=3):
        super().__init__()
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout) for _ in range(num_cmt_layers_second)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv_high_level):
        kv_high_level = self.norm(kv_high_level)
        for cross_attn in self.cross_attn_layers:
            q = cross_attn(q, kv_high_level)
        return q


class GatedFusion(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.attention_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feature1, feature2):
        combined_features = torch.cat([feature1, feature2], dim=-1)
        gate_feature = self.attention_gate(combined_features)
        return gate_feature


class MSCI(ThreeBranch):
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
        self.lamda_2 = nn.Parameter(torch.ones(output_dim) * self.clip_cfg.init_lamda)
        self.patch_norm = nn.LayerNorm(output_dim)
        # msci specific
        self.concat_projection_low = nn.Sequential(
            nn.Linear(self.clip_cfg.selected_low_layers * output_dim, output_dim),
            nn.LayerNorm(output_dim), nn.ReLU(), nn.Dropout(0.1)
        )
        self.concat_projection_high = nn.Sequential(
            nn.Linear(self.clip_cfg.selected_high_layers * output_dim, output_dim),
            nn.LayerNorm(output_dim), nn.ReLU(), nn.Dropout(0.1)
        )
        self.stage1_local_alignment = Stage1LocalAlignment(d_model=output_dim,
                                                           num_heads=self.clip_cfg.stage_1_num_heads,
                                                           dropout=self.clip_cfg.stage_1_dropout,
                                                           num_cmt_layers_first=self.clip_cfg.stage_1_num_cmt_layers)
        self.stage2_global_fusion = Stage2GlobalFusion(d_model=output_dim,
                                                       num_heads=self.clip_cfg.stage_2_num_heads,
                                                       dropout=self.clip_cfg.stage_2_dropout,
                                                       num_cmt_layers_second=self.clip_cfg.stage_2_num_cmt_layers)
        self.fusion_module = GatedFusion(output_dim, dropout=self.clip_cfg.fusion_dropout)

    def multi_stage_cross_attention(self, q, low_level_features, high_level_features):
        q_1 = self.stage1_local_alignment(q, low_level_features)
        q_2 = self.stage2_global_fusion(q_1, high_level_features)
        return q_1, q_2

    def aggregate_features_low(self, visual_features):
        assert isinstance(visual_features, list) and len(
            visual_features) > 0, "Input should be a non-empty list of tensors."
        batch_size, seq_len, feature_dim = visual_features[0].shape
        num_selected_layers = len(visual_features)
        concat_features = torch.cat(visual_features, dim=-1)  # [batch_size, seq_len, num_selected_layers * feature_dim]
        aggregated_features = self.concat_projection_low(concat_features)  # [batch_size, seq_len, feature_dim]
        return aggregated_features

    def aggregate_features_high(self, visual_features):
        assert isinstance(visual_features, list) and len(
            visual_features) > 0, "Input should be a non-empty list of tensors."

        batch_size, seq_len, feature_dim = visual_features[0].shape
        num_selected_layers = len(visual_features)

        concat_features = torch.cat(visual_features, dim=-1)  # [batch_size, seq_len, num_selected_layers * feature_dim]
        aggregated_features = self.concat_projection_high(concat_features)  # [batch_size, seq_len, feature_dim]
        return aggregated_features

    def encode_image_with_adapter(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) +
                       torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)
        x = x.permute(1, 0, 2)

        low_level_features = []
        mid_level_features = []
        high_level_features = []

        for i_block in range(self.clip.visual.transformer.layers):
            # MHA
            adapt_x = self.peft_tuner[i_block](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].attention(
                self.clip.visual.transformer.resblocks[i_block].ln_1(x)
            )
            x = x + adapt_x + residual
            # FFN
            i_adapter = i_block + self.clip.visual.transformer.layers
            adapt_x = self.peft_tuner[i_adapter](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].mlp(
                self.clip.visual.transformer.resblocks[i_block].ln_2(x)
            )
            x = x + adapt_x + residual
            # 保存每一层的特征
            x_feature = x.permute(1, 0, 2)  # LND -> NLD
            x_feature = self.clip.visual.ln_post(x_feature)
            if self.clip.visual.proj is not None:
                x_feature = x_feature @ self.clip.visual.proj  # [batch_size, seq_len, feature_dim]

            # 提取特定层的特征
            if i_block < self.clip_cfg.selected_low_layers:  # Low-level features (前8个Transformer块)
                low_level_features.append(x_feature)
            elif self.clip_cfg.selected_low_layers <= i_block < 24 - self.clip_cfg.selected_high_layers:  # Mid-level features (中间8个Transformer块)
                mid_level_features.append(x_feature)
            else:  # High-level features (最后8个Transformer块)
                high_level_features.append(x_feature)

        img_feature = high_level_features[-1]
        if self.clip_cfg.selected_high_layers == 1:
            high_level_features = high_level_features[-1]

        else:
            high_level_features = self.aggregate_features_high(high_level_features)
        if self.clip_cfg.selected_low_layers == 1:
            low_level_features = low_level_features[0]
        else:
            low_level_features = self.aggregate_features_low(low_level_features)
        return img_feature[:, 0, :], low_level_features, high_level_features

    def loss_calu(self, predict, target, idx):
        loss_fn = CrossEntropyLoss()
        _, batch_attr, batch_obj, batch_target = target
        if isinstance(predict, tuple) and len(predict) == 2:
            logits, (high_features, low_features, text_features) = predict
            comp_logits, attr_logits, obj_logits = logits
        else:
            logits = predict
            comp_logits, attr_logits, obj_logits = logits
            high_features = None
            low_features = None
            text_features = None

        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()

        # 计算分类损失
        loss_comp = loss_fn(comp_logits, batch_target)
        loss_attr = loss_fn(attr_logits, batch_attr)
        loss_obj = loss_fn(obj_logits, batch_obj)
        loss = loss_comp * self.config.pair_loss_weight + \
               loss_attr * self.config.attr_loss_weight + \
               loss_obj * self.config.obj_loss_weight
        if self.training:
            return {
                'loss': loss,
                'loss_c': loss_comp,
                'loss_a': loss_attr,
                'loss_o': loss_obj,
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

    def forward(self, batch, idx, return_features=False):
        batch_img, batch_attr, batch_obj, batch_pair = batch
        batch_img = batch_img.to(self.device)
        b = batch_img.shape[0]
        batch_img, low_level_features, high_level_features = self.encode_image_with_adapter(batch_img.type(self.clip.dtype))
        logits, text_feats, visual_feats = list(), dict(), dict()
        attr_feat, obj_feat = self.attr_disentangler(batch_img), self.obj_disentangler(batch_img)
        visual_feats['pair'] = batch_img / batch_img.norm(dim=-1, keepdim=True)
        visual_feats['obj'] = obj_feat / obj_feat.norm(dim=-1, keepdim=True)
        visual_feats['attr'] = attr_feat / attr_feat.norm(dim=-1, keepdim=True)
        if self.text_prompt_mode == "soft":
            text_feats = self._encode_comp_text_soft(idx)
        logits = list()
        for stage in ['pair', 'attr', 'obj']:
            # CMT
            idx_text_features = text_feats[stage]
            cmt_text_features = idx_text_features.unsqueeze(0).expand(b, -1, -1)
            cmt_1, cmt_2 = self.multi_stage_cross_attention(cmt_text_features, low_level_features, high_level_features)
            cmt_text_features = idx_text_features + self.lamda * cmt_1.squeeze(1) + self.lamda_2 * cmt_2.squeeze(1)
            cmt_text_features = cmt_text_features / cmt_text_features.norm(dim=-1, keepdim=True)
            logits.append(
                torch.einsum(
                    "bd, bkd->bk",
                    visual_feats[stage],
                    cmt_text_features * self.clip.logit_scale.exp()
                ))
        if return_features:
            return logits, (high_level_features, low_level_features, idx_text_features)
        else:
            return logits