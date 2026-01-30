import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from types import SimpleNamespace
from clip_modules.clip_model import load_clip
from clip_modules.tokenization_clip import SimpleTokenizer
from model.imagenet_template import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from model_ori.common import CustomTextEncoder
from model_ori.adapter import Adapter


class CustomCLIP(nn.Module):
    def __init__(self, config, attributes, classes, offset, device, logger):
        super().__init__()
        self.device = device
        self.clip_cfg = SimpleNamespace(**config.clip_config)
        self.clip = load_clip(
            name=self.clip_cfg.clip_model, context_length=config.context_length,
            device=device).to(self.device)

        self.tokenizer = SimpleTokenizer()
        self.config = config

        self.attributes, self.classes = attributes, classes

        self.visual_finetune = self.clip_cfg.visual_finetune
        self.text_prompt_mode = self.clip_cfg.text_prompt_mode
        logger.info(f'SET TEXT PROMPT MODE: {self.text_prompt_mode}')

        self.offset = offset
        self.num_layers = self.clip.visual.transformer.layers
        self.enable_pos_emb = True

        dtype = self.clip.dtype
        self.dtype = torch.bfloat16 if dtype is None else dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.tokenizer, self.config, self.dtype, logger)

        # frozen clip's encoder
        for p in self.parameters():
            p.requires_grad = False

        if self.visual_finetune:
            self.peft_tuner = self._set_peft_tuner()
        self._set_prompt_temp()

    def _set_peft_tuner(self):
        if self.clip_cfg.peft_tuner == 'AdapterFormer':
            adapter_num = 2 * self.clip.visual.transformer.layers
            peft_tuner = nn.ModuleList([
                Adapter(
                    d_model=self.clip.visual.transformer.width,
                    bottleneck=self.clip_cfg.adapter_dim,
                    dropout=self.clip_cfg.adapter_dropout,
                )
                for _ in range(adapter_num)
            ])
        else:
            raise ValueError(f"Unknown finetuner: {self.config.peft_tuner}")
        return peft_tuner

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
            token_ids, soft_att_obj, comp_ctx_vectors = self.construct_soft_prompt()
            self.attr_dropout = nn.Dropout(getattr(self.clip_cfg, "attr_dropout", 0.0))
            self.token_ids = token_ids
            self.soft_att_obj = nn.Parameter(soft_att_obj)
            self.comp_ctx_vectors = nn.Parameter(comp_ctx_vectors).to(self.device)
        else:
            raise ValueError(f"Unknown text_prompt_mode: {self.text_prompt_mode}")

    def encode_image(self, x: torch.Tensor):
        if self.visual_finetune:
            return self.encode_image_with_adapter(x)
        else:
            return self.encode_image_ori(x)

    def encode_image_with_adapter(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)                       # [B, C, H, W]
        x = x.reshape(x.shape[0], x.shape[1], -1)           # [B, C, HW]
        x = x.permute(0, 2, 1)                              # [B, HW, C]
        x = torch.cat(
            [self.clip.visual.class_embedding.to(x.dtype)
             + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1, )  # [B, HW+1, C]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)                              # [L, B, C]
        feat_list = []
        for i_block in range(self.clip.visual.transformer.layers):
            # MHA + Adapter
            adapt_x = self.peft_tuner[i_block](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].attention(
                self.clip.visual.transformer.resblocks[i_block].ln_1(x)
            )
            x = x + adapt_x + residual
            # FFN + Adapter
            i_adapter = i_block + self.clip.visual.transformer.layers
            adapt_x = self.peft_tuner[i_adapter](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].mlp(
                self.clip.visual.transformer.resblocks[i_block].ln_2(x)
            )
            x = x + adapt_x + residual

            feat_list.append(x.permute(1, 0, 2))            # [B, L, C]

        img_feature = x.permute(1, 0, 2)                    # [B, L, C]
        img_feature = self.clip.visual.ln_post(img_feature)
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj

        return img_feature

    def encode_image_ori(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)                       # [B, C, H, W]
        x = x.reshape(x.shape[0], x.shape[1], -1)           # [B, C, HW]
        x = x.permute(0, 2, 1)                              # [B, HW, C]
        x = torch.cat(
            [self.clip.visual.class_embedding.to(x.dtype)
             + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1,) # [B, HW+1, C]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)                              # [L, B, C]
        feat_list = []
        for i_block in range(self.clip.visual.transformer.layers):
            x = self.clip.visual.transformer.resblocks[i_block](x)
            feat_list.append(x.permute(1, 0, 2))            # [B, L, C]
        img_feature = x.permute(1, 0, 2)                    # [B, L, C]
        img_feature = self.clip.visual.ln_post(img_feature)
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj

        return img_feature

    def encode_text(self, token_ids, token_tensors=None, enable_pos_emb=False):
        return self.text_encoder(token_ids, token_tensors, enable_pos_emb)

    def construct_soft_prompt(self):
        token_ids = self.tokenizer(
            self.clip_cfg.prompt_template, context_length=self.config.context_length).to(self.device)
        tokenized = torch.cat(
            [
                self.tokenizer(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.to(self.device))
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
            device=self.device,
            dtype=orig_token_embedding.dtype,
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], dim=0)

        ctx_init = self.clip_cfg.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(
            ctx_init,
            context_length=self.config.context_length
        ).to(self.device)
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)
        comp_ctx_vectors = embedding[0, 1: 1 + n_ctx[0], :].to(self.clip.dtype)

        return token_ids, soft_att_obj, comp_ctx_vectors

    def construct_token_tensors(self, pair_idx):
        if not hasattr(self, "token_ids") or self.token_ids is None:
            raise RuntimeError("construct_token_tensors only available when text_prompt_mode='soft'.")

        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        token_tensor, num_elements = [], [len(pair_idx), self.offset, len(self.classes)]

        for i_element in range(self.token_ids.shape[0]):
            class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element], 1)
            token_tensor.append(
                self.clip.token_embedding(
                    class_token_ids.to(self.device)
                ).type(self.clip.dtype)
            )

        eos_idx = [int(self.token_ids[i_element].argmax()) for i_element in range(self.token_ids.shape[0])]
        soft_att_obj = self.attr_dropout(self.soft_att_obj)

        token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[attr_idx].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[obj_idx + self.offset].type(self.clip.dtype)
        token_tensor[0][:, 1: len(self.comp_ctx_vectors) + 1, :] = self.comp_ctx_vectors.type(self.clip.dtype)

        return token_tensor

    def _encode_comp_text_soft(self, idx):
        token_tensors = self.construct_token_tensors(idx)
        i_element = 0
        _text_features, _ = self.encode_text(
            self.token_ids[i_element],
            token_tensors[i_element],
            enable_pos_emb=self.enable_pos_emb
        )
        comp_text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)
        return comp_text_features

    @staticmethod
    def _format_template(template: str, attr_name: str, obj_name: str) -> str:
        if ("{attr}" in template) or ("{obj}" in template):
            return template.format(attr=attr_name, obj=obj_name)
        else:
            label = f"{attr_name} {obj_name}"
            return template.format(label)

    def _encode_comp_text(self, idx):
        if self.prompt_templates is None:
            raise RuntimeError("prompt_templates is None")
        idx_list = idx.tolist() if isinstance(idx, torch.Tensor) else idx
        all_features = []
        for temp in self.prompt_templates:
            texts = []
            for a_idx, o_idx in idx_list:
                a_name = self.attributes[a_idx]
                o_name = self.classes[o_idx]
                text = self._format_template(temp, a_name, o_name)
                texts.append(text)
            token_ids = self.tokenizer(
                texts, context_length=self.config.context_length).to(self.device)
            text_features = self.clip.encode_text(token_ids)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_features.append(text_features)
        if len(all_features) == 1:
            comp_text_features = all_features[0]
        else:
            comp_text_features = torch.stack(all_features, dim=0).mean(dim=0)
        comp_text_features = comp_text_features / comp_text_features.norm(dim=-1, keepdim=True)
        return comp_text_features

    def loss_calu(self, comp_logits, target, idx):
        loss_fn = CrossEntropyLoss()
        _, _, _, batch_pair = target
        batch_pair = batch_pair.to(self.device)
        loss = loss_fn(comp_logits, batch_pair)
        if self.training:
            return {
                'loss_total': loss,
            }
        else:
            return loss

    def logit_infer(self, comp_logits, idx):
        return comp_logits

    def encode_text_for_open(self, idx):
        if self.text_prompt_mode == "soft":
            token_tensors = self.construct_token_tensors(idx)
            text_features = []
            for i_element in range(self.token_ids.shape[0]):
                _text_features, _ = self.encode_text(
                    self.token_ids[i_element],
                    token_tensors[i_element],
                    enable_pos_emb=self.enable_pos_emb,
                )
                idx_text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)
                text_features.append(idx_text_features)
            return text_features
        else:
            comp_features = self._encode_comp_text_clip_baseline(idx)
            return [comp_features]

    def forward(self, batch, idx):
        batch_img = batch[0].to(self.device)
        batch_img = self.encode_image(batch_img.type(self.clip.dtype))
        cls_token, patch_token = batch_img[:, 0, :], batch_img[:, 1:, :]
        if self.text_prompt_mode == "soft":
            comp_text_features = self._encode_comp_text_soft(idx)
        else:
            comp_text_features = self._encode_comp_text(idx)
        comp_text_features = comp_text_features / comp_text_features.norm(dim=-1, keepdim=True)
        cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
        logits = torch.einsum(
            "bd,kd->bk",
            cls_token, comp_text_features * self.clip.logit_scale.exp())

        return logits
