import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from model.baseline.baseline_siglip import CustomSigLIP
from model.utils.common import Disentangler

class ThreeBranchSigLIP(CustomSigLIP):
    def __init__(self, config, attributes, classes, offset, device, logger):
        super().__init__(config, attributes, classes, offset, device, logger)

        self.idx_mapping = {'pair': 0, 'attr': 1, 'obj': 2}
        self.attr_disentangler = Disentangler(self.visual_width)
        self.obj_disentangler = Disentangler(self.visual_width)
        self.text_prompt_mode = getattr(self.config, "text_prompt_mode", "ensemble")
        self._set_prompt_temp(logger)


    def _set_prompt_temp(self, logger=None):
        attr_dropout_p = getattr(self.clip_cfg, "attr_dropout", 0.3)
        if self.text_prompt_mode == "single":
            single_temp = getattr(self.clip_cfg, "single_prompt_template", "a photo of {}")
            self.prompt_templates = [single_temp]
            self.token_ids = None
            self.soft_att_obj = None
            self.comp_ctx_vectors = None
            self.attr_ctx_vectors = None
            self.obj_ctx_vectors = None
            if logger is not None:
                logger.info(f"Text Prompt Mode: single -> {single_temp}")

        elif self.text_prompt_mode == "ensemble":
            self.token_ids = None
            self.soft_att_obj = None
            self.comp_ctx_vectors = None
            self.attr_ctx_vectors = None
            self.obj_ctx_vectors = None
            if logger is not None:
                logger.info(
                    f"Text Prompt Mode: ensemble ({len(self.prompt_templates)} templates)"
                )

        elif self.text_prompt_mode == "soft":
            self.prompt_templates = None
            (
                token_ids,
                soft_att_obj,
                comp_ctx_vectors,
                attr_ctx_vectors,
                obj_ctx_vectors,
            ) = self.construct_soft_prompt()
            self.token_ids = token_ids
            self.attr_dropout = nn.Dropout(attr_dropout_p)
            self.soft_att_obj = nn.Parameter(soft_att_obj.to(self.device))
            self.comp_ctx_vectors = nn.Parameter(comp_ctx_vectors.to(self.device))
            self.attr_ctx_vectors = nn.Parameter(attr_ctx_vectors.to(self.device))
            self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors.to(self.device))
            if logger is not None:
                logger.info(
                    f"Text Prompt Mode: soft (Dynamic Learnable Prefix, n_attr={len(self.attributes)}, n_obj={len(self.classes)})"
                )
        else:
            raise ValueError(f"Unknown text_prompt_mode: {self.text_prompt_mode}")


    def construct_soft_prompt(self):
        device = self.device
        raw_token_embedding = self.clip.text_model.embeddings.token_embedding
        prompt_templates = self.clip_cfg.prompt_template
        context_length = self.config.context_length
        tok_temp = self.tokenizer(
            prompt_templates,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=context_length,
        )
        token_ids = tok_temp["input_ids"].to(device)
        all_words = self.attributes + self.classes
        tok_words = self.tokenizer(
            all_words,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=context_length
        ).to(device)

        with torch.no_grad():
            word_embeds = raw_token_embedding(tok_words.input_ids)

        soft_att_obj = torch.zeros(
            (len(all_words), word_embeds.size(-1)), device=device, dtype=word_embeds.dtype)

        eos_id = getattr(self.tokenizer, "eos_token_id", 1)
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)

        start_idx = 0
        for idx in range(len(all_words)):
            curr_ids = tok_words.input_ids[idx]
            eos_pos = (curr_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                end_idx = eos_pos[0].item()
            else:
                non_pad = (curr_ids != pad_id).nonzero(as_tuple=True)[0]
                end_idx = non_pad[-1].item() + 1 if len(non_pad) > 0 else len(curr_ids)
            if end_idx > start_idx:
                soft_att_obj[idx, :] = torch.mean(word_embeds[idx, start_idx:end_idx, :], dim=0)
            else:
                soft_att_obj[idx, :] = word_embeds[idx, start_idx, :]
        ctx_init = self.clip_cfg.ctx_init
        tok_ctx = self.tokenizer(
            ctx_init,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=context_length
        ).to(device)
        with torch.no_grad():
            ctx_embeds = raw_token_embedding(tok_ctx.input_ids)

        def get_len(text):
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            return len(ids)

        comp_ctx_len = get_len(ctx_init[0])
        attr_ctx_len = get_len(ctx_init[1])
        obj_ctx_len = get_len(ctx_init[2])
        comp_ctx_vectors = ctx_embeds[0, 0: comp_ctx_len, :].clone()
        attr_ctx_vectors = ctx_embeds[1, 0: attr_ctx_len, :].clone()
        obj_ctx_vectors = ctx_embeds[2, 0: obj_ctx_len, :].clone()
        return token_ids, soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors


    def construct_token_tensors(self, pair_idx):
        device = self.device
        raw_embedding_layer = self.clip.text_model.embeddings.token_embedding
        full_embeddings_module = self.clip.text_model.embeddings
        attr_idx, obj_idx = pair_idx[:, 0].to(device), pair_idx[:, 1].to(device)
        inputs_embeds_list = []
        num_elements = [pair_idx.size(0), self.offset, len(self.classes)]
        for i_element in range(self.token_ids.size(0)):
            base_ids = self.token_ids[i_element].unsqueeze(0).repeat(num_elements[i_element], 1)
            base_ids = base_ids.to(device)
            with torch.no_grad():
                embeds = raw_embedding_layer(base_ids)
            inputs_embeds_list.append(embeds)

        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", 1)
        eos_idx_list = []
        for i_element in range(self.token_ids.size(0)):
            ids = self.token_ids[i_element]
            if (ids == eos_token_id).any():
                pos = (ids == eos_token_id).nonzero(as_tuple=False)[-1, 0].item()
            else:
                pos = (ids != pad_id).nonzero(as_tuple=False)[-1, 0].item()
            eos_idx_list.append(int(pos))

        soft_att_obj = self.attr_dropout(self.soft_att_obj).to(device)

        idx_0 = eos_idx_list[0]
        inputs_embeds_list[0][:, idx_0 - 2, :] = soft_att_obj[attr_idx].type(self.dtype)
        inputs_embeds_list[0][:, idx_0 - 1, :] = soft_att_obj[obj_idx + self.offset].type(self.dtype)
        L0 = self.comp_ctx_vectors.size(0)
        inputs_embeds_list[0][:, 0: L0, :] = self.comp_ctx_vectors.type(self.dtype)
        idx_1 = eos_idx_list[1]
        inputs_embeds_list[1][:, idx_1 - 1, :] = soft_att_obj[: self.offset].type(self.dtype)
        L1 = self.attr_ctx_vectors.size(0)
        inputs_embeds_list[1][:, 0: L1, :] = self.attr_ctx_vectors.type(self.dtype)
        idx_2 = eos_idx_list[2]
        inputs_embeds_list[2][:, idx_2 - 1, :] = soft_att_obj[self.offset:].type(self.dtype)
        L2 = self.obj_ctx_vectors.size(0)
        inputs_embeds_list[2][:, 0: L2, :] = self.obj_ctx_vectors.type(self.dtype)
        final_token_tensors = []
        for i, raw_embeds in enumerate(inputs_embeds_list):
            final_embeds = full_embeddings_module(inputs_embeds=raw_embeds)
            final_token_tensors.append(final_embeds)
        return final_token_tensors


    def _render_template(self, template: str, content, is_attr) -> str:
        if "{}" in template:
            if isinstance(content, (list, tuple)):
                text = f"{content[0]} {content[1]}"
            else:
                if is_attr:
                    text = f"{content} object"
                else:
                    text = str(content)
            return template.format(text)
        if isinstance(content, (list, tuple)):
            return template.format(f"{content[0]} {content[1]}")
        if is_attr:
            return template.format(f"{content} object")
        return template.format(str(content))


    def _encode_list_with_templates(self, text_list, is_attr: bool = False):
        device = self.device
        all_feats = []
        TEXT_BATCH_SIZE = 32

        for temp in self.prompt_templates:
            prompts = [self._render_template(temp, text, is_attr) for text in text_list]
            template_feats_list = []
            for i in range(0, len(prompts), TEXT_BATCH_SIZE):
                batch_prompts = prompts[i: i + TEXT_BATCH_SIZE]
                tok = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=getattr(self.config, "context_length", 77),
                ).to(device)
                batch_feat = self.clip.get_text_features(**tok)
                template_feats_list.append(batch_feat)
            if len(template_feats_list) > 0:
                feats = torch.cat(template_feats_list, dim=0)
            else:
                feats = torch.zeros(0, self.visual_width, device=device)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats)

        if len(all_feats) > 1:
            final_feats = torch.stack(all_feats, dim=0).mean(dim=0)
        else:
            final_feats = all_feats[0]
        final_feats = final_feats / final_feats.norm(dim=-1, keepdim=True)
        return final_feats


    def _encode_comp_text(self, idx):
        if isinstance(idx, torch.Tensor):
            idx_list = idx.tolist()
        else:
            idx_list = idx
        pairs_text = [(self.attributes[a_i], self.classes[o_i]) for a_i, o_i in idx_list]
        return self._encode_list_with_templates(pairs_text, is_attr=False)


    def _encode_comp_text_soft(self, idx):
        token_tensors = self.construct_token_tensors(idx)
        text_feats, text_backbone = dict(), self.clip.text_model
        for stage, i_element in self.idx_mapping.items():
            embeds = token_tensors[i_element].to(self.device).to(self.dtype)
            encoder_outputs = text_backbone.encoder(
                inputs_embeds=embeds,
                attention_mask=None, output_attentions=False,
                output_hidden_states=False, return_dict=False,
            )
            last_hidden = encoder_outputs[0]
            last_hidden = text_backbone.final_layer_norm(last_hidden)
            pooled = last_hidden[:, -1, :]
            pooled = text_backbone.head(pooled)
            pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-7)   # 7767 413 674  for close-world
            text_feats[stage] = pooled
        return text_feats


    def encode_text_for_open(self, pairs):
        token_tensors = self.construct_token_tensors(pairs)
        text_feats, text_backbone = {}, self.clip.text_model
        batch_size = self.config.text_encoder_batch_size
        def text_encode_func(text_embed):
            encoder_outputs = text_backbone.encoder(
                inputs_embeds=text_embed, attention_mask=None, output_attentions=False, output_hidden_states=False,
                return_dict=False, )
            pooled = text_backbone.final_layer_norm(encoder_outputs[0])[:, -1, :]
            pooled = text_backbone.head(pooled)
            pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-7)
            return pooled
        for stage, i_element in self.idx_mapping.items():
            embeds = token_tensors[i_element].to(self.device).to(self.dtype)
            chunks = [text_encode_func(embeds[i: i + batch_size]) for i in range(0, embeds.size(0), batch_size)]
            text_feats[stage] = torch.cat(chunks, 0) if len(chunks) > 1 else chunks[0]
        return text_feats


    def encode_text(self, idx):
        if self.text_prompt_mode == "soft":
            text_feat_dict = self._encode_comp_text_soft(idx)
        else:
            text_feat_dict = {
                "pair": self._encode_comp_text(idx),
                "attr": self._encode_list_with_templates(self.attributes, is_attr=True),
                "obj": self._encode_list_with_templates(self.classes, is_attr=False),
            }
        return text_feat_dict

    def loss_calu(self, logits, target, idx):
        loss_fn = CrossEntropyLoss()
        _, batch_attr, batch_obj, batch_pair = target
        pair_logits, attr_logits, obj_logits = logits

        batch_attr = batch_attr.to(self.device)
        batch_obj = batch_obj.to(self.device)
        batch_pair = batch_pair.to(self.device)
        loss_pair = loss_fn(pair_logits, batch_pair) * self.config.pair_loss_weight
        loss_attr = loss_fn(attr_logits, batch_attr) * self.config.prim_loss_weight
        loss_obj = loss_fn(obj_logits, batch_obj) * self.config.prim_loss_weight
        loss = loss_pair + loss_attr + loss_obj

        if self.training:
            loss_dict = {
                'loss': loss,
                'loss_pair': loss_pair,
                'loss_prim': loss_attr + loss_obj,
            }
            return loss_dict
        else:
            return loss

    def logit_infer(self, logits, idx):
        pair_logits, attr_logits, obj_logits = logits
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)
        primitive_weight, pair_weight = self.config.primitive_inference_weight, self.config.pair_inference_weight
        for i_comp in range(pair_logits.shape[-1]):
            a_idx, o_idx = idx[i_comp, 0], idx[i_comp, 1]
            weighted_attr_pred = attr_pred[:, a_idx] * primitive_weight
            weighted_obj_pred = obj_pred[:, o_idx] * primitive_weight
            pair_logits[:, i_comp] = (
                    pair_logits[:, i_comp] * pair_weight
                    + weighted_attr_pred * weighted_obj_pred
            )
        return pair_logits

    def forward(self, batch, idx):
        batch_img = batch[0].to(self.device)
        img_features = self.encode_image(batch_img)  # [B, D]
        cls_token = img_features
        attr_feat = self.attr_disentangler(cls_token)
        obj_feat = self.obj_disentangler(cls_token)
        visual_feat_dict = {
            "pair": cls_token / cls_token.norm(dim=-1, keepdim=True),
            "attr": attr_feat / attr_feat.norm(dim=-1, keepdim=True),
            "obj": obj_feat / obj_feat.norm(dim=-1, keepdim=True),
        }
        text_feat_dict = self.encode_text(idx)
        logits = list()
        for stage in ["pair", "attr", "obj"]:
            stage_logits = torch.einsum(
                "bd,kd->bk",
                visual_feat_dict[stage], text_feat_dict[stage], ) * self.clip.logit_scale.exp().clamp(max=100)
            logits.append(stage_logits)
        return logits