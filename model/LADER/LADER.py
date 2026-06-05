import torch
import torch.nn as nn
import torch.nn.functional as F
from model.baseline.ThreeBranch import ThreeBranch
from model.utils.losses import compute_conditional_orthogonality, MaxSup
from model.utils.common_utils import compute_base_logits
from model.LADER.utils import (
    ConfigAccessor,
    ContextBoundedManifoldDecoder,
    build_active_mask,
    pool_query_groups,
    split_query_groups,
)


class LADER(ThreeBranch):
    def __init__(self, config, attributes, classes, offset, device, logger):
        super().__init__(config, attributes, classes, offset, device, logger)
        del self.attr_disentangler, self.obj_disentangler
        vision_width = self.clip.visual.transformer.width
        self.vision_width = vision_width
        output_dim = self.clip.visual.output_dim
        cfg = ConfigAccessor(self.clip_cfg, self.config)
        ctx_query_count = cfg.int(('num_q_ctx', 'q_ctx_num', 'q_ctx_count', 'q_ctx'), 1)
        self.query_sizes = (
            cfg.int(('num_q_attr', 'q_attr_num', 'q_attr_count', 'q_attr'), 1),
            cfg.int(('num_q_obj', 'q_obj_num', 'q_obj_count', 'q_obj'), 1),
            ctx_query_count,
        )
        self.inject_layer_idx = cfg.nonnegative_int(('inject_layer_idx', 'query_inject_layer'), 0)
        self.query_pooling = str(cfg.value(('query_pooling', 'q_pooling'), 'similarity')).lower()
        if self.query_pooling not in {'similarity', 'mean', 'first', 'max'}:
            raise ValueError(f"Unknown query_pooling mode: {self.query_pooling}")
        self.allow_primitive_cls_attention = cfg.bool(('allow_primitive_cls_attention',), False)
        self.q_attr = nn.Parameter(torch.randn(self.query_sizes[0], 1, vision_width) * 0.02)
        self.q_obj = nn.Parameter(torch.randn(self.query_sizes[1], 1, vision_width) * 0.02)
        self.q_ctx = nn.Parameter(torch.randn(self.query_sizes[2], 1, vision_width) * 0.02)

        self.comp_decoder = self._build_comp_decoder(output_dim, cfg)

        use_maxsup = self.config.maxsup_smooth
        self.criterion = nn.CrossEntropyLoss() if not use_maxsup else MaxSup(self.config.smoothing)

    def _build_comp_decoder(self, output_dim, cfg):
        return ContextBoundedManifoldDecoder(
            dim=output_dim,
            hidden_dim=cfg.value(('decoder_hidden_dim',), output_dim),
            depth=cfg.value(('decoder_depth',), 2),
            mlp_ratio=cfg.value(('decoder_mlp_ratio',), 2.0),
            dropout=cfg.value(('decoder_dropout',), 0.1),
            use_transport=cfg.bool(('decoder_use_transport', 'use_transport_decoder'), True),
            transport_temperature=cfg.value(('decoder_transport_temperature',), 0.07),
            max_residual_scale=cfg.value(('decoder_max_residual_scale',), 0.35),
            normalize_output=cfg.bool(('decoder_normalize_output',), False),
        )

    def build_active_mask(self, L, device):
        return build_active_mask(
            L,
            self.query_sizes,
            device,
            allow_primitive_cls_attention=self.allow_primitive_cls_attention,
        )

    def _query_tokens(self, batch_size, device, dtype):
        queries = (self.q_attr, self.q_obj, self.q_ctx)
        return torch.cat([
            q.expand(-1, batch_size, -1).to(device=device, dtype=dtype)
            for q in queries
        ], dim=0)

    def _image_tokens(self, x):
        x = self.clip.visual.conv1(x)
        _, _, H_grid, W_grid = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls = self.clip.visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)

        pos_embed = self.clip.visual.positional_embedding
        if x.shape[1] != pos_embed.shape[0]:
            pos_embed = self.resize_pos_embed(pos_embed, H_grid, W_grid)
        x = x + pos_embed.to(x.dtype)
        x = self.clip.visual.ln_pre(x)
        return x.permute(1, 0, 2)  # [L, B, C]

    def encode_image_with_adapter(self, x: torch.Tensor, return_query_groups: bool = False):
        x = self._image_tokens(x)
        L, B, _ = x.shape
        active_mask = self.build_active_mask(L, x.device).to(x.dtype)
        current_seq = x
        for i_block in range(self.clip.visual.transformer.layers):
            block = self.clip.visual.transformer.resblocks[i_block]
            if i_block == self.inject_layer_idx:
                current_seq = torch.cat([current_seq, self._query_tokens(B, x.device, x.dtype)], dim=0)
            attn_mask = active_mask if i_block >= self.inject_layer_idx else None
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
        q_features = self.clip.visual.ln_post(final_seq[:, L:, :])
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
            q_features = q_features @ self.clip.visual.proj
        q_a, q_o, q_c = split_query_groups(q_features, self.query_sizes)
        z_a, z_o, z_c = pool_query_groups(
            q_features,
            self.query_sizes,
            self.query_pooling,
            groups=(q_a, q_o, q_c),
        )
        if not return_query_groups:
            return img_feature, z_a, z_o, z_c
        return img_feature, z_a, z_o, z_c, q_a, q_o

    def _decode_composition(self, z_a, z_o, z_c, pair_text, q_a=None, q_o=None):
        comp_feats = self.comp_decoder(z_a, z_o, z_c, q_a=q_a, q_o=q_o)
        comp_visual = F.normalize(comp_feats, dim=-1)
        comp_logits = self.clip.logit_scale.exp() * comp_visual @ pair_text.t()
        return comp_visual, comp_logits

    def loss_calu(self, logits, target, idx):
        loss_fn = self.criterion
        _, batch_attr, batch_obj, batch_pair = target
        pair_logits, attr_logits, obj_logits, comp_logits, loss_co = logits
        batch_attr = batch_attr.to(self.device)
        batch_obj = batch_obj.to(self.device)
        batch_pair = batch_pair.to(self.device)

        loss_comp = loss_fn(comp_logits, batch_pair) * self.config.comp_loss_weight
        loss_pair = loss_fn(pair_logits, batch_pair) * self.config.pair_loss_weight
        loss_attr = loss_fn(attr_logits, batch_attr) * self.config.attr_loss_weight
        loss_obj = loss_fn(obj_logits, batch_obj) * self.config.obj_loss_weight

        loss = loss_comp + loss_attr + loss_obj + loss_pair + loss_co
        if self.training:
            return {
                'loss': loss,
                'comp': loss_comp,
                'prim': loss_attr + loss_obj,
                'pair': loss_pair,
                'coc': loss_co,
            }
        return loss

    def logit_infer(self, logits, idx):
        pair_logits, attr_logits, obj_logits, comp_logits, _ = logits
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)

        for i_comp in range(pair_logits.shape[-1]):
            a_idx, o_idx = idx[i_comp, 0], idx[i_comp, 1]
            weighted_attr_pred = attr_pred[:, a_idx] * self.config.prim_inference_weight
            weighted_obj_pred = obj_pred[:, o_idx] * self.config.prim_inference_weight
            comp_term = comp_logits[:, i_comp] * self.config.comp_inference_weight
            pair_logits[:, i_comp] = (
                    pair_logits[:, i_comp] * self.config.pair_inference_weight +
                    comp_term +
                    weighted_attr_pred * weighted_obj_pred
            )
        return pair_logits

    def forward_for_open(self, batch, text_feature):
        batch_img = batch[0]
        batch_img = batch_img.to(self.device)
        img_feature, z_a, z_o, z_c, q_a, q_o = self.encode_image_with_adapter(
            batch_img.type(self.clip.dtype),
            return_query_groups=True,
        )

        logits, visual_feats = list(), dict()
        cls_token = img_feature[:, 0, :]
        visual_feats['pair'] = F.normalize(cls_token, dim=-1)
        visual_feats['obj'] = F.normalize(z_o, dim=-1)
        visual_feats['attr'] = F.normalize(z_a, dim=-1)
        logits = compute_base_logits(visual_feats, text_feature, self.clip.logit_scale)
        _, comp_logits = self._decode_composition(z_a, z_o, z_c, text_feature['pair'], q_a=q_a, q_o=q_o)
        logits.append(comp_logits)
        loss_co = compute_conditional_orthogonality(z_a, z_o, z_c) * self.config.cond_orth_loss_weight
        logits.append(loss_co)

        return logits

    def forward(self, batch, idx):
        batch_img = batch[0]
        batch_img = batch_img.to(self.device)
        img_feature, z_a, z_o, z_c, q_a, q_o = self.encode_image_with_adapter(
            batch_img.type(self.clip.dtype),
            return_query_groups=True,
        )

        logits, visual_feats = list(), dict()
        cls_token = img_feature[:, 0, :]
        visual_feats['pair'] = F.normalize(cls_token, dim=-1)
        visual_feats['obj'] = F.normalize(z_o, dim=-1)
        visual_feats['attr'] = F.normalize(z_a, dim=-1)
        text_feats = self._encode_comp_text_soft(idx)
        logits = compute_base_logits(visual_feats, text_feats, self.clip.logit_scale)

        _, comp_logits = self._decode_composition(z_a, z_o, z_c, text_feats['pair'], q_a=q_a, q_o=q_o)
        logits.append(comp_logits)

        loss_co = compute_conditional_orthogonality(z_a, z_o, z_c) * self.config.cond_orth_loss_weight
        logits.append(loss_co)

        return logits
