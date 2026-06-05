from types import SimpleNamespace
from typing import List
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoTokenizer, SiglipModel
from peft import LoraConfig, get_peft_model
from model.thirdparty.adapter import Adapter
from model.utils.imagenet_template import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from typing import Optional

SINGLE_TEMPLATES = ["a photo of {}."]


class CustomSigLIPImageModel(nn.Module):
    def __init__(self, vision_model, clip_cfg, logger):
        super(CustomSigLIPImageModel, self).__init__()
        self.clip_cfg = clip_cfg
        self.peft_mode = clip_cfg.visual_peft_mode
        self.adapter_dim = clip_cfg.lora_r_visual
        self.adapter_dropout = clip_cfg.lora_dropout
        self.embeddings = vision_model.embeddings
        self.encoder_layers = vision_model.encoder.layers
        self.num_layers = len(self.encoder_layers)
        self.embed_dim = vision_model.encoder.layers[0].embed_dim
        self.post_layernorm = vision_model.post_layernorm
        self.use_head = vision_model.use_head
        if self.use_head:
            self.head = vision_model.head
        for param in self.parameters():
            param.requires_grad = False
        self.add_visual_tunable_params(logger)

    def add_visual_tunable_params(self, logger):
        if self.peft_mode in ['AdaptFormer', ...]:
            adapter_num = 2 * self.num_layers
            self.adapter = nn.ModuleList([Adapter(d_model=self.embed_dim,
                                            bottleneck=self.adapter_dim,
                                            dropout=self.adapter_dropout,
                                            ) for _ in range(adapter_num)])
            logger.info(f'>>> Applying AdaptFormer in Visual Side... Adapter BottleNeck is {self.adapter_dim}, Num: {adapter_num}')

        elif self.peft_mode == 'Lora':
            target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
            peft_config = LoraConfig(
                r=self.adapter_dim,
                lora_alpha=2 * self.adapter_dim,
                target_modules=target_modules,
                lora_dropout=self.adapter_dropout,
                bias="none",
                init_lora_weights='gaussian'
            )
            self.encoder_layers = get_peft_model(self.encoder_layers, peft_config)
            logger.info(f'Applying Lora in Visual Side... Lora Rank is {self.adapter_dim}')
            logger.info('Visual Lora Param setting...')
            self.encoder_layers.print_trainable_parameters()

    def forward_for_adapter(self, hidden_states):
        for idx in range(self.num_layers):
            # MHA
            attn_block = self.encoder_layers[idx].self_attn
            layer_norm1 = self.encoder_layers[idx].layer_norm1
            adapt_x = self.adapter[idx](hidden_states, add_residual=False)
            residual = hidden_states
            hidden_states = layer_norm1(hidden_states)
            hidden_states, attn_weights = attn_block(
                hidden_states=hidden_states,
                attention_mask=None,
                output_attentions=False,
            )
            hidden_states = residual + hidden_states + adapt_x
            # FFN
            ffn_block = self.encoder_layers[idx].mlp
            layer_norm2 = self.encoder_layers[idx].layer_norm2
            i_adapter = idx + self.num_layers
            adapt_x = self.adapter[i_adapter](hidden_states, add_residual=False)
            residual = hidden_states
            hidden_states = layer_norm2(hidden_states)
            hidden_states = ffn_block(hidden_states)
            hidden_states = residual + hidden_states + adapt_x
            # record
        last_hidden_state = self.post_layernorm(hidden_states)
        pooler_output = self.head(last_hidden_state)
        return [pooler_output, last_hidden_state]

    def forward_for_lora(self, hidden_states):
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states)
        last_hidden_state = self.post_layernorm(hidden_states)
        pooler_output = self.head(last_hidden_state)
        return [pooler_output, last_hidden_state]

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=False)
        if self.peft_mode == 'AdaptFormer':
            output = self.forward_for_adapter(hidden_states)
        elif self.peft_mode == 'Lora':
            output = self.forward_for_lora(hidden_states)
        return output


class CustomSigLIPTextModel(nn.Module):
    def __init__(self, text_model):
        super(CustomSigLIPTextModel, self).__init__()
        self.embeddings = text_model.embeddings
        self.encoder_layers = text_model.encoder.layers
        self.num_layers = len(self.encoder_layers)
        self.embed_dim = text_model.config.hidden_size
        self.final_layer_norm = text_model.final_layer_norm
        self.head = text_model.head
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input_ids,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states=hidden_states,
                                          attention_mask=attention_mask,)
        last_hidden_state = hidden_states[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        # Assuming "sticky" EOS tokenization, last token is always EOS.
        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.head(pooled_output)
        return pooled_output


class CustomSigLIP(nn.Module):
    def __init__(self, config, attributes: List[str], classes: List[str], offset, device, logger):
        super().__init__()
        self.device = device
        self.config = config
        self.clip_cfg = SimpleNamespace(**config.clip_config)
        self.attributes, self.classes = list(attributes), list(classes)
        self.num_attrs, self.num_classes = len(self.attributes), len(self.classes)
        self.offset = offset
        model_name = getattr(self.clip_cfg, "hf_model_name", None)
        if model_name is None:
            model_name = self.clip_cfg.clip_arch
        logger.info(f">>> Loading HF Vision-Language Model: {model_name}...")

        self.image_mean = tuple(getattr(self.config, "image_mean", getattr(self.clip_cfg, "input_image_mean", (0.5, 0.5, 0.5))))
        self.image_std = tuple(getattr(self.config, "image_std", getattr(self.clip_cfg, "input_image_std", (0.5, 0.5, 0.5))))
        self.clip_cfg.input_image_mean = getattr(self.clip_cfg, "input_image_mean", self.image_mean)
        self.clip_cfg.input_image_std = getattr(self.clip_cfg, "input_image_std", self.image_std)

        attn_impl = getattr(self.clip_cfg, "attn_implementation", "sdpa")
        local_files_only = getattr(self.clip_cfg, "local_files_only", False)
        self.clip = SiglipModel.from_pretrained(
            model_name,
            attn_implementation=attn_impl,
            local_files_only=local_files_only,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.num_layers = self.clip.config.vision_config.num_hidden_layers
        self.visual_width = self.clip.config.vision_config.hidden_size

        self.dtype = next(self.clip.parameters()).dtype
        for p in self.clip.parameters():
            p.requires_grad = False
        self.logger = logger
        self._setup_finetuning()
        self.prompt_templates = SINGLE_TEMPLATES

    def _setup_finetuning(self):
        visual_ft = self.clip_cfg.visual_finetune
        text_ft = self.clip_cfg.text_finetune
        if self.clip_cfg.text_peft_mode == 'Lora' or self.clip_cfg.visual_peft_mode == 'Lora':
            target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
            self.logger.info(f'>>> Lora Default target is {target_modules}')
        if visual_ft:
            self.clip.vision_model = CustomSigLIPImageModel(self.clip.vision_model, self.clip_cfg, self.logger)
        if text_ft:
            if self.clip_cfg.text_peft_mode == 'Lora':
                # self.clip.text_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                if self.config.gradient_checkpointing_enable:
                    self.clip.text_model.gradient_checkpointing = True
                peft_config = LoraConfig(
                    r=self.clip_cfg.lora_r_text,
                    lora_alpha=2 * self.clip_cfg.lora_r_text,
                    target_modules=target_modules,
                    lora_dropout=self.clip_cfg.lora_dropout,
                    bias="none", init_lora_weights='gaussian')
                self.logger.info(f">>> Applying LoRA in Text Side, Lora Rank is {self.clip_cfg.lora_r_text}")
                self.clip.text_model = get_peft_model(self.clip.text_model, peft_config)
                self.logger.info('>>> Text Lora Param setting...')
                self.clip.text_model.print_trainable_parameters()
            else:
                raise ValueError("text finetuning mode not supported.")

    def _render_template(self, template: str, attr: str, cls: str) -> str:
        if "{attr}" in template or "{cls}" in template:
            return template.format(attr=attr, cls=cls)
        if "{}" in template:
            n = template.count("{}")
            if n == 2:
                return template.format(attr, cls)
            elif n == 1:
                return template.format(f"{attr} {cls}")
        return f"{template} {attr} {cls}"

    def _encode_comp_text(self, idx) -> torch.Tensor:
        device = self.device
        if isinstance(idx, torch.Tensor):
            idx_list = idx.tolist()
        else:
            idx_list = idx
        current_attrs = [self.attributes[a_i] for a_i, o_i in idx_list]
        current_objs = [self.classes[o_i] for a_i, o_i in idx_list]
        all_feats = []
        TEXT_BATCH_SIZE = 32
        for temp in self.prompt_templates:
            prompts = [
                self._render_template(temp, attr, obj)
                for attr, obj in zip(current_attrs, current_objs)
            ]
            template_feats_list = []
            for i in range(0, len(prompts), TEXT_BATCH_SIZE):
                batch_prompts = prompts[i: i + TEXT_BATCH_SIZE]
                tok = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
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
            text_features = torch.stack(all_feats, dim=0).mean(dim=0)
        else:
            text_features = all_feats[0]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device).to(self.dtype)
        image_features = self.clip.vision_model(pixel_values=x)
        return image_features

    def loss_calu(self, comp_logits, target, idx):
        loss_fn = nn.CrossEntropyLoss()
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

    def forward(self, batch, idx):
        batch_img = batch[0].to(self.device)
        cls_token = self.encode_image(batch_img)  # [B, D]
        cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
        comp_text_features = self._encode_comp_text(idx)  # [K, D]
        logit_scale = self.clip.logit_scale.exp()
        logits = torch.einsum("bd,kd->bk", cls_token, comp_text_features) * logit_scale

        return logits


class SigLIP_ZeroShot_Baseline(nn.Module):
    """Frozen SigLIP zero-shot CZSL baseline with single or ensemble prompts."""

    def __init__(self, config, attributes: List[str], classes: List[str], offset, device, logger):
        super().__init__()
        self.device = device
        self.config = config
        self.clip_cfg = SimpleNamespace(**config.clip_config)
        self.attributes, self.classes = list(attributes), list(classes)
        self.num_attrs, self.num_classes = len(self.attributes), len(self.classes)
        self.offset = offset
        self.logger = logger

        model_name = getattr(self.clip_cfg, "hf_model_name", None)
        if model_name is None:
            model_name = getattr(self.clip_cfg, "clip_arch", None)
        if model_name is None:
            raise ValueError("SigLIP_ZeroShot_Baseline needs clip_config.hf_model_name or clip_config.clip_arch.")

        attn_impl = getattr(self.clip_cfg, "attn_implementation", "sdpa")
        local_files_only = getattr(self.clip_cfg, "local_files_only", False)
        logger.info(f">>> Loading HF SigLIP Zero-Shot Model: {model_name}...")
        self.clip = SiglipModel.from_pretrained(
            model_name,
            attn_implementation=attn_impl,
            local_files_only=local_files_only,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            use_fast=False,
        )
        self.dtype = next(self.clip.parameters()).dtype
        self.visual_width = self._feature_dim()
        self.text_batch_size = getattr(config, "text_encoder_batch_size", 256)
        self.text_max_length = self._resolve_text_max_length()
        self.text_prompt_mode = getattr(config, "text_prompt_mode", getattr(self.clip_cfg, "text_prompt_mode", "single"))
        self.image_already_siglip_processed = getattr(self.clip_cfg, "image_already_siglip_processed", False)
        self._setup_image_normalization()

        for param in self.clip.parameters():
            param.requires_grad = False

        self._set_prompt_templates(logger)

    def _setup_image_normalization(self):
        input_mean = getattr(self.clip_cfg, "input_image_mean", getattr(self.config, "image_mean", (0.5, 0.5, 0.5)))
        input_std = getattr(self.clip_cfg, "input_image_std", getattr(self.config, "image_std", (0.5, 0.5, 0.5)))
        siglip_mean = getattr(self.image_processor, "image_mean", (0.5, 0.5, 0.5))
        siglip_std = getattr(self.image_processor, "image_std", (0.5, 0.5, 0.5))
        self.register_buffer("input_image_mean", self._image_stats_tensor(input_mean), persistent=False)
        self.register_buffer("input_image_std", self._image_stats_tensor(input_std), persistent=False)
        self.register_buffer("siglip_image_mean", self._image_stats_tensor(siglip_mean), persistent=False)
        self.register_buffer("siglip_image_std", self._image_stats_tensor(siglip_std), persistent=False)

    @staticmethod
    def _image_stats_tensor(value):
        tensor = torch.tensor(value, dtype=torch.float32)
        if tensor.numel() == 1:
            tensor = tensor.repeat(3)
        return tensor.view(1, 3, 1, 1)

    def _feature_dim(self) -> int:
        text_cfg = getattr(self.clip.config, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "projection_size"):
            return text_cfg.projection_size
        if hasattr(self.clip.config, "projection_dim"):
            return self.clip.config.projection_dim
        return self.clip.config.vision_config.hidden_size

    def _resolve_text_max_length(self) -> int:
        cfg_len = getattr(self.clip_cfg, "text_max_length", None)
        if cfg_len is not None:
            return int(cfg_len)

        tokenizer_len = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_len, int) and 0 < tokenizer_len < 10000:
            return tokenizer_len

        text_cfg = getattr(self.clip.config, "text_config", None)
        model_len = getattr(text_cfg, "max_position_embeddings", None)
        if model_len is not None:
            return int(model_len)
        return 64

    def _set_prompt_templates(self, logger=None):
        if self.text_prompt_mode == "single":
            single_template = getattr(self.clip_cfg, "single_prompt_template", SINGLE_TEMPLATES[0])
            self.prompt_templates = [single_template]
            if logger is not None:
                logger.info(f"Text Prompt Mode: single -> {single_template}")
        elif self.text_prompt_mode == "ensemble":
            ensemble_type = getattr(self.clip_cfg, "ensemble_type", "select")
            self.prompt_templates = IMAGENET_TEMPLATES if ensemble_type == "imagenet" else IMAGENET_TEMPLATES_SELECT
            if logger is not None:
                logger.info(f"Text Prompt Mode: ensemble ({len(self.prompt_templates)} templates, type={ensemble_type})")
        else:
            raise ValueError("SigLIP_ZeroShot_Baseline only supports text_prompt_mode in ['single', 'ensemble'].")

    def _render_template(self, template: str, attr: str, cls: str) -> str:
        label = f"{attr} {cls}".strip()
        if "{label}" in template or "{attr}" in template or "{cls}" in template or "{obj}" in template:
            return template.format(label=label, attr=attr, cls=cls, obj=cls)
        if "{}" in template:
            n_slots = template.count("{}")
            if n_slots == 2:
                return template.format(attr, cls)
            if n_slots == 1:
                return template.format(label)
        return f"{template} {label}"

    @staticmethod
    def _unwrap_features(features):
        if torch.is_tensor(features):
            return features
        if hasattr(features, "pooler_output"):
            return features.pooler_output
        if isinstance(features, (tuple, list)):
            return features[0]
        raise TypeError(f"Unsupported SigLIP feature output type: {type(features)}")

    def _encode_text_prompts(self, prompts: List[str]) -> torch.Tensor:
        feature_chunks = []
        for start in range(0, len(prompts), self.text_batch_size):
            batch_prompts = prompts[start: start + self.text_batch_size]
            tokenized = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.text_max_length,
            ).to(self.device)
            text_features = self._unwrap_features(self.clip.get_text_features(**tokenized))
            feature_chunks.append(text_features)

        if len(feature_chunks) == 0:
            return torch.zeros(0, self.visual_width, device=self.device, dtype=self.dtype)
        text_features = torch.cat(feature_chunks, dim=0)
        return text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-7)

    def _encode_comp_text(self, idx) -> torch.Tensor:
        idx_list = idx.tolist() if isinstance(idx, torch.Tensor) else idx
        current_attrs = [self.attributes[a_i] for a_i, _ in idx_list]
        current_objs = [self.classes[o_i] for _, o_i in idx_list]

        all_template_features = []
        for template in self.prompt_templates:
            prompts = [
                self._render_template(template, attr, obj)
                for attr, obj in zip(current_attrs, current_objs)
            ]
            all_template_features.append(self._encode_text_prompts(prompts))

        if len(all_template_features) == 1:
            text_features = all_template_features[0]
        else:
            text_features = torch.stack(all_template_features, dim=0).mean(dim=0)
        return text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-7)

    def encode_text_for_open(self, idx):
        return self._encode_comp_text(idx)

    def _preprocess_image(self, x: torch.Tensor) -> torch.Tensor:
        if self.image_already_siglip_processed:
            return x
        input_mean = self.input_image_mean.to(device=x.device, dtype=x.dtype)
        input_std = self.input_image_std.to(device=x.device, dtype=x.dtype)
        siglip_mean = self.siglip_image_mean.to(device=x.device, dtype=x.dtype)
        siglip_std = self.siglip_image_std.to(device=x.device, dtype=x.dtype)
        x = x * input_std + input_mean
        x = x.clamp(0.0, 1.0)
        return (x - siglip_mean) / siglip_std

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device).to(self.dtype)
        x = self._preprocess_image(x)
        image_features = self._unwrap_features(self.clip.get_image_features(pixel_values=x))
        return image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-7)

    def _logits_from_features(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("bd,kd->bk", image_features, text_features)
        logits = logits * self.clip.logit_scale.exp()
        return logits

    def loss_calu(self, comp_logits, target, idx):
        loss_fn = nn.CrossEntropyLoss()
        _, _, _, batch_pair = target
        batch_pair = batch_pair.to(self.device)
        loss = loss_fn(comp_logits, batch_pair)
        if self.training:
            return {
                "loss": loss,
                "loss_total": loss,
            }
        return loss

    def logit_infer(self, comp_logits, idx):
        return comp_logits

    def forward_for_open(self, batch, text_feats):
        batch_img = batch[0].to(self.device)
        image_features = self.encode_image(batch_img)
        return self._logits_from_features(image_features, text_feats)

    def forward(self, batch, idx):
        batch_img = batch[0].to(self.device)
        image_features = self.encode_image(batch_img)
        text_features = self._encode_comp_text(idx)
        return self._logits_from_features(image_features, text_features)
