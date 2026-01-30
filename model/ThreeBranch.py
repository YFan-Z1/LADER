import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from model.baseline import CustomCLIP
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from model.common import Disentangler
from model.imagenet_template import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

class ThreeBranch(CustomCLIP):
    def __init__(self, config, attributes, classes, offset, device, logger):
        super(ThreeBranch, self).__init__(config, attributes, classes, offset, device, logger)
        self.idx_mapping = {'pair': 0, 'attr': 1, 'obj': 2}
        self.reverse_mapping = {v: k for k, v in self.idx_mapping.items()}
        self.attr_disentangler = Disentangler(self.clip.visual.output_dim)
        self.obj_disentangler = Disentangler(self.clip.visual.output_dim)
        self._set_prompt_temp()

    def _set_prompt_temp(self):
        if self.text_prompt_mode == 'single':
            self.prompt_templates = [self.clip_cfg.single_prompt_template]
            self.token_ids = None
            self.soft_att_obj = None
            self.comp_ctx_vectors = None

        elif self.text_prompt_mode == 'ensemble':
            self.prompt_templates = IMAGENET_TEMPLATES if self.clip_cfg.ensemble_type == 'imagenet' else IMAGENET_TEMPLATES_SELECT
            self.token_ids = None
            self.soft_att_obj = None
            self.comp_ctx_vectors = None

        elif self.text_prompt_mode == 'soft':
            self.prompt_templates = None
            self.token_ids, soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors = self.construct_soft_prompt()
            self.attr_dropout = nn.Dropout(getattr(self.clip_cfg, "attr_dropout", 0.3))
            self.soft_att_obj = nn.Parameter(soft_att_obj)
            self.comp_ctx_vectors = nn.Parameter(comp_ctx_vectors).to(self.device)
            self.attr_ctx_vectors = nn.Parameter(attr_ctx_vectors).to(self.device)
            self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors).to(self.device)
        else:
            raise ValueError(f"Unknown text_prompt_mode: {self.text_prompt_mode}")

    def construct_soft_prompt(self):
        # token_ids indicates the position of [EOS]
        token_ids = self.tokenizer(self.clip_cfg.prompt_template,
                                   context_length=self.config.context_length).to(self.device)
        tokenized = torch.cat(
            [
                self.tokenizer(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.to(self.device))
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = self.clip_cfg.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init,
                                context_length=self.config.context_length).to(self.device)
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx_vectors = embedding[0, 1: 1 + n_ctx[0], :].to(self.clip.dtype)  # comp context
        attr_ctx_vectors = embedding[1, 1: 1 + n_ctx[1], :].to(self.clip.dtype)  # attr context
        obj_ctx_vectors = embedding[2, 1: 1 + n_ctx[2], :].to(self.clip.dtype)  # obj context

        return token_ids, soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors

    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        token_tensor, num_elements = list(), [len(pair_idx), self.offset, len(self.classes)]
        for i_element in range(self.token_ids.shape[0]):
            class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element], 1)
            token_tensor.append(self.clip.token_embedding(
                class_token_ids.to(self.device)
            ).type(self.clip.dtype))
        eos_idx = [int(self.token_ids[i_element].argmax()) for i_element in range(self.token_ids.shape[0])]
        soft_att_obj = self.attr_dropout(self.soft_att_obj)

        # pair
        token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[attr_idx].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[obj_idx + self.offset].type(self.clip.dtype)
        token_tensor[0][:, 1: len(self.comp_ctx_vectors) + 1, :] = self.comp_ctx_vectors.type(self.clip.dtype)
        # attr
        token_tensor[1][:, eos_idx[1] - 1, :] = soft_att_obj[:self.offset].type(self.clip.dtype)
        token_tensor[1][:, 1: len(self.attr_ctx_vectors) + 1, :] = self.attr_ctx_vectors.type(self.clip.dtype)
        # obj
        token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[self.offset:].type(self.clip.dtype)
        token_tensor[2][:, 1: len(self.obj_ctx_vectors) + 1, :] = self.obj_ctx_vectors.type(self.clip.dtype)

        return token_tensor

    def _encode_comp_text_soft(self, idx):
        token_tensors = self.construct_token_tensors(idx)
        text_feats = {}
        for stage in self.idx_mapping:
            i_element = self.idx_mapping[stage]
            _text_features, _ = self.encode_text(
                self.token_ids[i_element], token_tensors[i_element], enable_pos_emb=self.enable_pos_emb
            )
            text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)
            text_feats[stage] = text_features
        return text_feats


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

        loss = loss_pair + loss_attr + loss_obj

        if self.training:
            loss_dict = {
                'loss': loss,
                'loss_pair': loss_pair,
                'loss_attr': loss_attr,
                'loss_obj': loss_obj,
            }
            return loss_dict
        else:
            return loss


    def logit_infer(self, logits, idx):
        pair_logits, attr_logits, obj_logits = logits
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)
        for i_comp in range(pair_logits.shape[-1]):
            a_idx, o_idx = idx[i_comp, 0], idx[i_comp, 1]
            weighted_attr_pred = 1.0 if self.config.attr_inference_weight == 0 \
                else attr_pred[:, a_idx] * self.config.attr_inference_weight
            weighted_obj_pred = 1.0 if self.config.obj_inference_weight == 0 \
                else obj_pred[:, o_idx] * self.config.obj_inference_weight
            pair_logits[:, i_comp] = (
                    pair_logits[:, i_comp] * self.config.pair_inference_weight
                    + weighted_attr_pred * weighted_obj_pred
            )

        return pair_logits

    def forward(self, batch, idx):
        batch_img = batch[0].to(self.device)
        batch_img = self.encode_image(batch_img.type(self.clip.dtype))
        cls_token, patch_token = batch_img[:, 0, :], batch_img[:, 1:, :]
        attr_feat, obj_feat = self.attr_disentangler(cls_token), self.obj_disentangler(cls_token)
        visual_feat_dict, logits = dict(), list()
        visual_feat_dict['pair'] = cls_token / cls_token.norm(dim=-1, keepdim=True)
        visual_feat_dict['attr'] = attr_feat / attr_feat.norm(dim=-1, keepdim=True)
        visual_feat_dict['obj'] = obj_feat / obj_feat.norm(dim=-1, keepdim=True)
        text_feat_dict = self._encode_comp_text_soft(idx)
        for stage in self.idx_mapping.keys():
            logits.append(torch.einsum(
                "bd,kd->bk",
                visual_feat_dict[stage], text_feat_dict[stage] * self.clip.logit_scale.exp()))
        return logits