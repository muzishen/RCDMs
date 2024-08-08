from dataclasses import dataclass
from typing import Dict, Optional, Union


from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.configuration_utils import ConfigMixin, register_to_config

from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import BaseOutput
from .attention import BasicTransformerBlock
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from .motion_module import get_motion_module
import os, json
@dataclass
class PriorTransformerOutput(BaseOutput):
    """
    The output of [`PriorTransformer`].

    Args:
        predicted_image_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`):
            The predicted CLIP image embedding conditioned on the CLIP text embedding input.
    """

    predicted_image_embedding: torch.FloatTensor

class MyPriorTransformer(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    """
    A Prior Transformer model.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 32): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_layers (`int`, *optional*, defaults to 20): The number of layers of Transformer blocks to use.
        embedding_dim (`int`, *optional*, defaults to 768): The dimension of the model input `hidden_states`
        num_embeddings (`int`, *optional*, defaults to 77):
            The number of embeddings of the model input `hidden_states`
        additional_embeddings (`int`, *optional*, defaults to 4): The number of additional tokens appended to the
            projected `hidden_states`. The actual length of the used `hidden_states` is `num_embeddings +
            additional_embeddings`.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        time_embed_act_fn (`str`, *optional*, defaults to 'silu'):
            The activation function to use to create timestep embeddings.
        norm_in_type (`str`, *optional*, defaults to None): The normalization layer to apply on hidden states before
            passing to Transformer blocks. Set it to `None` if normalization is not needed.
        embedding_proj_norm_type (`str`, *optional*, defaults to None):
            The normalization layer to apply on the input `proj_embedding`. Set it to `None` if normalization is not
            needed.
        encoder_hid_proj_type (`str`, *optional*, defaults to `linear`):
            The projection layer to apply on the input `encoder_hidden_states`. Set it to `None` if
            `encoder_hidden_states` is `None`.
        added_emb_type (`str`, *optional*, defaults to `prd`): Additional embeddings to condition the model.
            Choose from `prd` or `None`. if choose `prd`, it will prepend a token indicating the (quantized) dot
            product between the text embedding and image embedding as proposed in the unclip paper
            https://arxiv.org/abs/2204.06125 If it is `None`, no additional embeddings will be prepended.
        time_embed_dim (`int, *optional*, defaults to None): The dimension of timestep embeddings.
            If None, will be set to `num_attention_heads * attention_head_dim`
        embedding_proj_dim (`int`, *optional*, default to None):
            The dimension of `proj_embedding`. If None, will be set to `embedding_dim`.
        clip_embed_dim (`int`, *optional*, default to None):
            The dimension of the output. If None, will be set to `embedding_dim`.
    """

    @register_to_config
    def __init__(
            self,
            num_attention_heads: int = 32,
            attention_head_dim: int = 64,
            num_layers: int = 20,
            embedding_dim: int = 768,
            num_embeddings=77,
            additional_embeddings=4,
            dropout: float = 0.0,
            time_embed_act_fn: str = "silu",
            norm_in_type: Optional[str] = None,  # layer
            embedding_proj_norm_type: Optional[str] = None,  # layer
            encoder_hid_proj_type: Optional[str] = "linear",  # linear
            added_emb_type: Optional[str] = "prd",  # prd
            time_embed_dim: Optional[int] = None,
            embedding_proj_dim: Optional[int] = None,
            clip_embed_dim: Optional[int] = None,

            unet_use_cross_frame_attention=None,
            unet_use_temporal_attention=None,

            use_motion_module=None,
            motion_module_type=None,
            motion_module_kwargs=None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.additional_embeddings = additional_embeddings

        time_embed_dim = time_embed_dim or inner_dim
        embedding_proj_dim = embedding_proj_dim or embedding_dim
        clip_embed_dim = clip_embed_dim or embedding_dim

        self.time_proj = Timesteps(inner_dim, True, 0)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, out_dim=inner_dim,
                                                act_fn=time_embed_act_fn)

        self.proj_in = nn.Linear(embedding_dim, inner_dim)

        if embedding_proj_norm_type is None:
            self.embedding_proj_norm = None
        elif embedding_proj_norm_type == "layer":
            self.embedding_proj_norm = nn.LayerNorm(embedding_proj_dim)
        else:
            raise ValueError(f"unsupported embedding_proj_norm_type: {embedding_proj_norm_type}")

        self.embedding_proj = nn.Linear(embedding_proj_dim, inner_dim)
        self.embedding_proj1 = nn.Linear(embedding_proj_dim, inner_dim)
        self.embedding_proj2 = nn.Linear(embedding_proj_dim, inner_dim)

        if encoder_hid_proj_type is None:
            self.encoder_hidden_states_proj = None
            self.encoder_hidden_states_proj1 = None
        elif encoder_hid_proj_type == "linear":
            self.encoder_hidden_states_proj = nn.Linear(embedding_dim, inner_dim)
            self.encoder_hidden_states_proj1 = nn.Linear(1664, inner_dim)
        else:
            raise ValueError(f"unsupported encoder_hid_proj_type: {encoder_hid_proj_type}")

        self.positional_embedding = nn.Parameter(torch.zeros(1, num_embeddings + additional_embeddings, inner_dim))

        if added_emb_type == "prd":
            self.prd_embedding = nn.Parameter(torch.zeros(1, 1, inner_dim))
        elif added_emb_type is None:
            self.prd_embedding = None
        else:
            raise ValueError(
                f"`added_emb_type`: {added_emb_type} is not supported. Make sure to choose one of `'prd'` or `None`."
            )

        from itertools import chain
        self.transformer_blocks = nn.ModuleList(
            list(chain(*[
                (
                    BasicTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        activation_fn="gelu",
                        attention_bias=True,
                        unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                        unet_use_temporal_attention=unet_use_temporal_attention,
                    ),
                    get_motion_module(
                        in_channels=inner_dim,
                        prior_state =True,
                        motion_module_type=motion_module_type,
                        motion_module_kwargs=motion_module_kwargs,
                    ) if use_motion_module else None
                )
                for d in range(num_layers)
            ]))
        )



        if norm_in_type == "layer":
            self.norm_in = nn.LayerNorm(inner_dim)
        elif norm_in_type is None:
            self.norm_in = None
        else:
            raise ValueError(f"Unsupported norm_in_type: {norm_in_type}.")

        self.norm_out = nn.LayerNorm(inner_dim)

        self.proj_to_clip_embeddings = nn.Linear(inner_dim, clip_embed_dim)

        causal_attention_mask = torch.full(
            [num_embeddings + additional_embeddings, num_embeddings + additional_embeddings], -10000.0
        )
        causal_attention_mask.triu_(1)
        causal_attention_mask = causal_attention_mask[None, ...]
        self.register_buffer("causal_attention_mask", causal_attention_mask, persistent=False)

        # self.clip_mean = nn.Parameter(torch.zeros(1, clip_embed_dim))
        # self.clip_std = nn.Parameter(torch.zeros(1, clip_embed_dim))
        self.clip_mean = torch.tensor(-0.016)
        self.clip_std = torch.tensor(0.415)
    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module,
                                        processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def forward(
            self,
            hidden_states,
            timestep: Union[torch.Tensor, float, int],
            proj_embedding: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            proj_embedding1: Optional[torch.FloatTensor] = None,
            mask_label: Optional[torch.FloatTensor] = None,
            # encoder_hidden_states1: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            return_dict: bool = True,
    ):
        """
        The [`PriorTransformer`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`):
                The currently predicted image embeddings.
            timestep (`torch.LongTensor`):
                Current denoising step.
            proj_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`):
                Projected embedding vector the denoising process is conditioned on.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_embeddings, embedding_dim)`):
                Hidden states of the text embeddings the denoising process is conditioned on.
            attention_mask (`torch.BoolTensor` of shape `(batch_size, num_embeddings)`):
                Text mask for the text embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.prior_transformer.PriorTransformerOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.prior_transformer.PriorTransformerOutput`] or `tuple`:
                If return_dict is True, a [`~models.prior_transformer.PriorTransformerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """

        batch_size = hidden_states.shape[0]

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=hidden_states.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(hidden_states.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(batch_size, dtype=timesteps.dtype, device=timesteps.device)

        timesteps_projected = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might be fp16, so we need to cast here.
        timesteps_projected = timesteps_projected.to(dtype=self.dtype)
        time_embeddings = self.time_embedding(timesteps_projected)

        if self.embedding_proj_norm is not None:

            proj_embedding = self.embedding_proj_norm(proj_embedding)

        proj_embeddings = self.embedding_proj(proj_embedding)
        proj_embeddings1 = self.embedding_proj1(proj_embedding1)
        masked_label_embeds = self.embedding_proj2(mask_label)

        if self.encoder_hidden_states_proj is not None and encoder_hidden_states is not None:

            encoder_hidden_states = self.encoder_hidden_states_proj(encoder_hidden_states)
            # encoder_hidden_states1 = self.encoder_hidden_states_proj1(encoder_hidden_states1)
        elif self.encoder_hidden_states_proj is not None and encoder_hidden_states is None:
            raise ValueError("`encoder_hidden_states_proj` requires `encoder_hidden_states` to be set")

        hidden_states = self.proj_in(hidden_states)

        positional_embeddings = self.positional_embedding.to(hidden_states.dtype)

        additional_embeds = []

        if encoder_hidden_states is not None:
            additional_embeds.append(encoder_hidden_states)


        if len(proj_embeddings.shape) == 2:
            proj_embeddings = proj_embeddings[:, None, :]

        if len(proj_embeddings1.shape) == 2:
            proj_embeddings1 = proj_embeddings1[:, None, :]

        if len(masked_label_embeds.shape) == 2:
            masked_label_embeds = masked_label_embeds[:, None, :]

        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states[:, None, :]

        additional_embeds = additional_embeds + [
            proj_embeddings,
            # encoder_hidden_states1,
            proj_embeddings1,
            masked_label_embeds,
            time_embeddings[:, None, :],
            hidden_states,
        ]
# 91 + 1 + 1 + 1 + 1 + 1  = 96
        if self.prd_embedding is not None:

            prd_embedding = self.prd_embedding.to(hidden_states.dtype).expand(batch_size, -1, -1)
            additional_embeds.append(prd_embedding)

        hidden_states = torch.cat(
            additional_embeds,
            dim=1,
        )
# 96 + 1


        hidden_states = hidden_states + positional_embeddings

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = F.pad(attention_mask, (0, self.additional_embeddings), value=0.0)
            attention_mask = (attention_mask[:, None, :] + self.causal_attention_mask).to(hidden_states.dtype)
            attention_mask = attention_mask.repeat_interleave(self.config.num_attention_heads, dim=0)

        if self.norm_in is not None:
            hidden_states = self.norm_in(hidden_states)
        # print(hidden_states.shape)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask) # bf, 95, 2048

        hidden_states = self.norm_out(hidden_states)

        hidden_states = hidden_states[:, -1]


        predicted_image_embedding = self.proj_to_clip_embeddings(hidden_states)

        if not return_dict:
            return (predicted_image_embedding,)

        return PriorTransformerOutput(predicted_image_embedding=predicted_image_embedding)

    def post_process_latents(self, prior_latents):
        prior_latents = (prior_latents * self.clip_std) + self.clip_mean
        return prior_latents
    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, unet_additional_kwargs=None):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded temporal prior's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)
        config["_class_name"] = cls.__name__
        config["num_embeddings"] = 91    # 85 /91
        config["additional_embeddings"] = 6

        from diffusers.utils import WEIGHTS_NAME
        model = cls.from_config(config, **unet_additional_kwargs)

        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        if not os.path.isfile(model_file):
            raise RuntimeError(f"{model_file} does not exist")
        state_dict = torch.load(model_file, map_location="cpu")

        # drop conv in
        match_dict = {k: v for k, v in state_dict.items() if not k.startswith("positional_embedding")}


        m, u = model.load_state_dict(match_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

        params = [p.numel() if "temporal" in n else 0 for n, p in model.named_parameters()]
        print(f"### Temporal Module Parameters: {sum(params) / 1e6} M")

        return model