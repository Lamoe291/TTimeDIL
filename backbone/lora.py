# Sheng Wang at Feb 22 2023

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from safetensors import safe_open
# from safetensors.torch import save_file
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
from torch.nn.parameter import Parameter

from backbone.base_vit import ViT
import os
from backbone.linears import SimpleLinear
import gc
import torch.nn.utils as utils
import copy


class AdapterModule(nn.Module):
    """A lightweight adapter block for ViT blocks."""

    def __init__(self, dim: int, bottleneck: int = 64, dropout: float = 0.1, scale: float = 1.0):
        super().__init__()
        self.down_proj = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck, dim, bias=False)
        self.scale = scale

        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.down_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up_proj(x) * self.scale
        return residual + x


class PromptModule(nn.Module):
    """Learnable prompt tokens prepended to the input sequence."""

    def __init__(self, embed_dim: int, prompt_length: int = 5):
        super().__init__()
        self.prompt_length = prompt_length
        self.prompt = nn.Parameter(torch.zeros(1, prompt_length, embed_dim))
        nn.init.normal_(self.prompt, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        prompts = self.prompt.expand(batch_size, -1, -1).to(x.device)
        return torch.cat([prompts, x], dim=1)


class _PEFTBlock(nn.Module):
    """Wraps a ViT block with optional prompt tokens and an adapter."""

    def __init__(self, block: nn.Module, embed_dim: int, prompt_length: int = 0, adapter_bottleneck: int = 64,
                 adapter_dropout: float = 0.1, adapter_scale: float = 1.0):
        super().__init__()
        self.block = block
        self.prompt = PromptModule(embed_dim, prompt_length) if prompt_length > 0 else None
        self.adapter = AdapterModule(embed_dim, bottleneck=adapter_bottleneck, dropout=adapter_dropout,
                                     scale=adapter_scale) if adapter_bottleneck > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        if self.prompt is not None:
            x = self.prompt(x)

        x = self.block(x)

        if self.prompt is not None:
            x = x[:, self.prompt.prompt_length:]

        if self.adapter is not None:
            x = self.adapter(x)

        return x


class PEFT_ViT_timm(nn.Module):
    """Wrapper for timm ViT models with optional adapters and prompts.

    Args:
        vit_model: a timm-style ViT model
        use_adapter: whether to insert lightweight adapters after each block
        adapter_bottleneck: bottleneck size for each adapter
        adapter_dropout: dropout applied inside adapters
        adapter_scale: residual scaling factor for adapter outputs
        use_prompt: whether to prepend learnable prompt tokens to each block
        prompt_length: number of prompt tokens per block
        freeze_backbone: whether to freeze the original backbone weights
    """

    def __init__(self, vit_model: timm_ViT, use_adapter: bool = False, adapter_bottleneck: int = 64,
                 adapter_dropout: float = 0.1, adapter_scale: float = 1.0, use_prompt: bool = False,
                 prompt_length: int = 5, freeze_backbone: bool = True):
        super().__init__()

        self.base_vit = copy.deepcopy(vit_model)
        self.vit_model = vit_model
        self.use_adapter = use_adapter
        self.use_prompt = use_prompt

        if freeze_backbone:
            for param in self.vit_model.parameters():
                param.requires_grad = False

        embed_dim = getattr(vit_model, "embed_dim", None)
        if embed_dim is None and hasattr(vit_model, "blocks") and len(vit_model.blocks) > 0:
            first_block = vit_model.blocks[0]
            embed_dim = getattr(first_block, "norm1", None)
            if embed_dim is not None and hasattr(embed_dim, "normalized_shape"):
                embed_dim = embed_dim.normalized_shape[0]
            else:
                embed_dim = first_block.attn.qkv.in_features

        if embed_dim is None:
            embed_dim = 768

        peft_blocks = []
        for block in vit_model.blocks:
            peft_blocks.append(_PEFTBlock(
                block=block,
                embed_dim=embed_dim,
                prompt_length=prompt_length if use_prompt else 0,
                adapter_bottleneck=adapter_bottleneck if use_adapter else 0,
                adapter_dropout=adapter_dropout,
                adapter_scale=adapter_scale,
            ))

        self.vit_model.blocks = nn.Sequential(*peft_blocks)

    def forward(self, x: Tensor):
        return self.vit_model(x)


class PrefixTuningModule(nn.Module):
    def __init__(self, prefix_dim: int, prefix_length: int = 5):
        super().__init__()
        self.prefix_length = prefix_length
        self.prefix_k = nn.Parameter(torch.zeros(1, prefix_length, prefix_dim))
        self.prefix_v = nn.Parameter(torch.zeros(1, prefix_length, prefix_dim))
        nn.init.normal_(self.prefix_k, mean=0.0, std=0.02)
        nn.init.normal_(self.prefix_v, mean=0.0, std=0.02)

    def get_prefixes(self, batch_size: int, device: torch.device):
        return (
            self.prefix_k.expand(batch_size, -1, -1).to(device),
            self.prefix_v.expand(batch_size, -1, -1).to(device),
        )


class _PrefixTunedAttention(nn.Module):
    def __init__(self, attn_module: nn.Module, prefix_modules: nn.ModuleList, current_task: int = 0):
        super().__init__()
        self.attn = attn_module
        self.prefix_modules = prefix_modules
        self.current_task = current_task
        self.task_weights = None
        self.top_k = None

    def set_current_task(self, task_id: int):
        self.current_task = task_id

    def set_task_weights(self, task_weights: torch.Tensor):
        self.task_weights = task_weights

    def _select_topk_weights(self, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.to(weights.device)
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        if self.top_k is None or self.top_k >= weights.shape[1]:
            return weights
        _, topk_indices = weights.topk(self.top_k, dim=1)
        mask = torch.zeros_like(weights)
        mask.scatter_(1, topk_indices, 1.0)
        return weights * mask

    def _get_weighted_prefixes(self, batch_size: int, device: torch.device):
        if self.prefix_modules is None or self.task_weights is None:
            return None, None

        prefix_k_tokens = torch.stack([m.prefix_k.squeeze(0) for m in self.prefix_modules], dim=0)
        prefix_v_tokens = torch.stack([m.prefix_v.squeeze(0) for m in self.prefix_modules], dim=0)
        weights = self.task_weights
        weights = weights.to(device)
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        weights = self._select_topk_weights(weights)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)
        weights = weights[:, :, None, None]
        combined_k = (weights * prefix_k_tokens.unsqueeze(0)).sum(dim=1)
        combined_v = (weights * prefix_v_tokens.unsqueeze(0)).sum(dim=1)
        return combined_k.expand(batch_size, -1, -1).to(device), combined_v.expand(batch_size, -1, -1).to(device)

    def _get_current_task_prefixes(self, batch_size: int, device: torch.device):
        if self.prefix_modules is None:
            return None, None
        module = self.prefix_modules[self.current_task]
        return module.get_prefixes(batch_size, device)

    def forward(self, x: Tensor, attn_mask=None, is_causal: bool = False):
        B, N, C = x.shape
        qkv = self.attn.qkv(x).reshape(B, N, 3, self.attn.num_heads, self.attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.attn.q_norm(q), self.attn.k_norm(k)

        prefix_k, prefix_v = self._get_weighted_prefixes(B, x.device)
        if prefix_k is None or prefix_v is None:
            prefix_k, prefix_v = self._get_current_task_prefixes(B, x.device)

        if prefix_k is not None and prefix_v is not None:
            prefix_k = prefix_k[:, None, :, :].expand(-1, self.attn.num_heads, -1, -1)
            prefix_v = prefix_v[:, None, :, :].expand(-1, self.attn.num_heads, -1, -1)
            k = torch.cat([prefix_k, k], dim=2)
            v = torch.cat([prefix_v, v], dim=2)

        if self.attn.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn.attn_drop.p if self.attn.training else 0.0,
                is_causal=is_causal,
            )
        else:
            q = q * self.attn.scale
            attn = q @ k.transpose(-2, -1)
            attn_bias = self.attn.resolve_self_attn_mask(N, attn, attn_mask, is_causal) if hasattr(self.attn, 'resolve_self_attn_mask') else None
            if attn_bias is not None:
                attn = maybe_add_mask(attn, attn_bias)
            attn = attn.softmax(dim=-1)
            attn = self.attn.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.attn.attn_dim)
        x = self.attn.norm(x)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x


class _TaskWeightedPEFTBlock(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        embed_dim: int,
        num_tasks: int = 1,
        current_task: int = 0,
        prompt_length: int = 0,
        adapter_bottleneck: int = 64,
        adapter_dropout: float = 0.1,
        adapter_scale: float = 1.0,
        top_k: int = None,
        init_from_previous_task: bool = False,
        use_prefix: bool = False,
        prefix_length: int = 5,
    ):
        super().__init__()
        self.block = block
        self.embed_dim = embed_dim
        self.prompt_length = prompt_length
        self.num_tasks = max(1, num_tasks)
        self.current_task = current_task
        self.task_weights = None
        self.top_k = top_k
        self.init_from_previous_task = init_from_previous_task
        self.use_prefix = use_prefix and prefix_length > 0
        self.prefix_length = prefix_length

        self.use_prompt = prompt_length > 0
        self.use_adapter = adapter_bottleneck > 0

        if self.use_prompt:
            self.prompt_modules = nn.ModuleList([
                PromptModule(embed_dim, prompt_length) for _ in range(self.num_tasks)
            ])
        else:
            self.prompt_modules = None

        if self.use_adapter:
            self.adapter_modules = nn.ModuleList([
                AdapterModule(embed_dim, bottleneck=adapter_bottleneck, dropout=adapter_dropout,
                              scale=adapter_scale)
                for _ in range(self.num_tasks)
            ])
        else:
            self.adapter_modules = None

        if self.use_prefix:
            attn_module = getattr(self.block, 'attn', None)
            prefix_dim = getattr(attn_module, 'head_dim', None)
            if prefix_dim is None:
                prefix_dim = embed_dim // max(1, getattr(attn_module, 'num_heads', 1))
            self.prefix_modules = nn.ModuleList([
                PrefixTuningModule(prefix_dim, prefix_length=self.prefix_length)
                for _ in range(self.num_tasks)
            ])
            self._wrap_attention()
        else:
            self.prefix_modules = None

        # Optionally initialize the new task's PEFT modules from the previous task
        if self.init_from_previous_task and self.current_task > 0:
            prev = self.current_task - 1
            if self.use_prompt and self.prompt_modules is not None and prev < len(self.prompt_modules) and self.current_task < len(self.prompt_modules):
                with torch.no_grad():
                    self.prompt_modules[self.current_task].prompt.data.copy_(
                        self.prompt_modules[prev].prompt.data
                    )
            if self.use_adapter and self.adapter_modules is not None and prev < len(self.adapter_modules) and self.current_task < len(self.adapter_modules):
                # copy adapter weights from previous task
                prev_state = self.adapter_modules[prev].state_dict()
                self.adapter_modules[self.current_task].load_state_dict({k: v.clone() for k, v in prev_state.items()})
            if self.use_prefix and self.prefix_modules is not None and prev < len(self.prefix_modules) and self.current_task < len(self.prefix_modules):
                with torch.no_grad():
                    self.prefix_modules[self.current_task].prefix_k.data.copy_(
                        self.prefix_modules[prev].prefix_k.data
                    )
                    self.prefix_modules[self.current_task].prefix_v.data.copy_(
                        self.prefix_modules[prev].prefix_v.data
                    )

        self._update_trainable_modules()

    def _update_trainable_modules(self):
        if self.prompt_modules is not None:
            for idx, module in enumerate(self.prompt_modules):
                for param in module.parameters():
                    param.requires_grad = idx == self.current_task
        if self.adapter_modules is not None:
            for idx, module in enumerate(self.adapter_modules):
                for param in module.parameters():
                    param.requires_grad = idx == self.current_task
        if self.prefix_modules is not None:
            for idx, module in enumerate(self.prefix_modules):
                for param in module.parameters():
                    param.requires_grad = idx == self.current_task

    def _wrap_attention(self):
        if not hasattr(self.block, 'attn'):
            return
        attn = getattr(self.block, 'attn', None)
        if isinstance(attn, _PrefixTunedAttention):
            attn.prefix_modules = self.prefix_modules
            attn.top_k = self.top_k
            return
        wrapped_attn = _PrefixTunedAttention(attn, self.prefix_modules, current_task=self.current_task)
        wrapped_attn.top_k = self.top_k
        self.block.attn = wrapped_attn

    def set_current_task(self, task_id: int):
        self.current_task = task_id
        self._update_trainable_modules()
        if self.prefix_modules is not None and hasattr(self.block, 'attn') and isinstance(self.block.attn, _PrefixTunedAttention):
            self.block.attn.set_current_task(task_id)

    def set_task_weights(self, task_weights: torch.Tensor):
        self.task_weights = task_weights
        if self.prefix_modules is not None and hasattr(self.block, 'attn') and isinstance(self.block.attn, _PrefixTunedAttention):
            self.block.attn.set_task_weights(task_weights)

    def _select_topk_weights(self, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.to(weights.device)
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        weights = weights[:, : self.num_tasks]
        if self.top_k is None or self.top_k >= weights.shape[1]:
            return weights

        topk_weights, topk_indices = weights.topk(self.top_k, dim=1)
        mask = torch.zeros_like(weights)
        mask.scatter_(1, topk_indices, 1.0)
        return weights * mask

    def _weighted_prompt(self, x: Tensor, weights: torch.Tensor) -> Tensor:
        prompt_tokens = torch.stack([m.prompt.squeeze(0) for m in self.prompt_modules], dim=0)
        # prompt_tokens: (T, prompt_length, embed_dim)
        weights = self._select_topk_weights(weights)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)
        weights = weights[:, :, None, None]
        combined = (weights * prompt_tokens.unsqueeze(0)).sum(dim=1)
        return torch.cat([combined, x], dim=1)

    def _weighted_adapter(self, x: Tensor, weights: torch.Tensor) -> Tensor:
        x_base = x
        weights = self._select_topk_weights(weights)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)
        residual = torch.zeros_like(x)
        for idx, adapter in enumerate(self.adapter_modules):
            w = weights[:, idx].view(-1, 1, 1)
            residual += w * (adapter(x_base) - x_base)
        return x_base + residual

    def forward(self, x: Tensor) -> Tensor:
        use_weighted_eval = not self.training and self.task_weights is not None
        if self.prompt_modules is not None:
            if use_weighted_eval:
                x = self._weighted_prompt(x, self.task_weights)
            else:
                prompts = self.prompt_modules[self.current_task].prompt.expand(x.shape[0], -1, -1).to(x.device)
                x = torch.cat([prompts, x], dim=1)

        x = self.block(x)

        if self.prompt_modules is not None:
            x = x[:, self.prompt_length:]

        if self.adapter_modules is not None:
            if use_weighted_eval:
                x = self._weighted_adapter(x, self.task_weights)
            else:
                x = self.adapter_modules[self.current_task](x)

        return x


class TaskWeightedPEFT_ViT_timm(nn.Module):
    def __init__(
        self,
        vit_model: timm_ViT,
        use_adapter: bool = False,
        adapter_bottleneck: int = 64,
        adapter_dropout: float = 0.1,
        adapter_scale: float = 1.0,
        use_prompt: bool = False,
        prompt_length: int = 5,
        num_tasks: int = 1,
        current_task: int = 0,
        top_k: int = None,
        freeze_backbone: bool = True,
        init_from_previous_task: bool = True,
        use_prefix: bool = False,
        prefix_length: int = 5,
    ):
        super().__init__()
        self.base_vit = copy.deepcopy(vit_model)
        self.vit_model = vit_model
        self.use_adapter = use_adapter
        self.use_prompt = use_prompt
        self.prompt_length = prompt_length
        self.num_tasks = max(1, num_tasks)
        self.current_task = current_task
        self.task_weights = None
        self.top_k = top_k
        self.init_from_previous_task = init_from_previous_task
        self.use_prefix = use_prefix and prefix_length > 0
        self.prefix_length = prefix_length

        if freeze_backbone:
            for param in self.vit_model.parameters():
                param.requires_grad = False

        embed_dim = getattr(vit_model, "embed_dim", None)
        if embed_dim is None and hasattr(vit_model, "blocks") and len(vit_model.blocks) > 0:
            first_block = vit_model.blocks[0]
            embed_dim = getattr(first_block, "norm1", None)
            if embed_dim is not None and hasattr(embed_dim, "normalized_shape"):
                embed_dim = embed_dim.normalized_shape[0]
            else:
                embed_dim = first_block.attn.qkv.in_features

        if embed_dim is None:
            embed_dim = 768

        wrapped_blocks = []
        for block in vit_model.blocks:
            wrapped_blocks.append(_TaskWeightedPEFTBlock(
                block=block,
                embed_dim=embed_dim,
                num_tasks=self.num_tasks,
                current_task=self.current_task,
                prompt_length=prompt_length if use_prompt else 0,
                adapter_bottleneck=adapter_bottleneck if use_adapter else 0,
                adapter_dropout=adapter_dropout,
                adapter_scale=adapter_scale,
                top_k=top_k,
                init_from_previous_task=self.init_from_previous_task,
                use_prefix=self.use_prefix,
                prefix_length=self.prefix_length,
            ))
        self.vit_model.blocks = nn.Sequential(*wrapped_blocks)

    def set_current_task(self, task_id: int):
        self.current_task = task_id
        for block in self.vit_model.blocks:
            block.set_current_task(task_id)

    def set_task_weights(self, task_weights: torch.Tensor):
        self.task_weights = task_weights
        for block in self.vit_model.blocks:
            block.set_task_weights(task_weights)

    def forward(self, x: Tensor, task_weights: torch.Tensor = None):
        if task_weights is not None:
            self.set_task_weights(task_weights)
        return self.vit_model(x)

    def _get_device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def activate_eval(self):
        self.eval()

    def deactivate_eval(self):
        self.train()

    def set_internal_tt_weights(self, weights: torch.Tensor):
        if weights is None:
            self.set_task_weights(None)
            return
        weights = weights.detach()
        if weights.device != self._get_device():
            weights = weights.to(self._get_device())
        self.set_task_weights(weights)

    def reset_internal_tt_weights(self, full_weight_task_id: int = 0):
        device = self._get_device()
        weights = torch.zeros(self.num_tasks, device=device, dtype=torch.float32)
        if 0 <= full_weight_task_id < self.num_tasks:
            weights[full_weight_task_id] = 1.0
        else:
            weights[0] = 1.0
        self.set_task_weights(weights)

    def load_peft_parameters(self, filename: str) -> None:
        """Load previously saved PEFT modules (prompts/adapters) from disk."""
        if not os.path.exists(filename):
            return

        device = self._get_device()
        for task_id in range(self.current_task):
            filepath = os.path.join(filename, f"peft_task_{task_id}.pt")
            if not os.path.exists(filepath):
                continue

            try:
                state = torch.load(filepath, map_location=device)

                if self.use_prompt:
                    if "prompts" in state:
                        for block, prompt in zip(self.vit_model.blocks, state["prompts"]):
                            block.prompt_modules[task_id].prompt.data = prompt.to(device)
                    elif f"prompt_{task_id}" in state:
                        for block in self.vit_model.blocks:
                            block.prompt_modules[task_id].prompt.data = state[f"prompt_{task_id}"].to(device)

                if self.use_adapter:
                    if "adapters" in state:
                        for block, adapter_state in zip(self.vit_model.blocks, state["adapters"]):
                            block.adapter_modules[task_id].load_state_dict(adapter_state)
                    elif f"adapter_{task_id}" in state:
                        for block in self.vit_model.blocks:
                            block.adapter_modules[task_id].load_state_dict(state[f"adapter_{task_id}"])

                if getattr(self, "use_prefix", False):
                    if "prefixes_k" in state:
                        for block, prefix_k in zip(self.vit_model.blocks, state["prefixes_k"]):
                            block.prefix_modules[task_id].prefix_k.data = prefix_k.to(device)
                    if "prefixes_v" in state:
                        for block, prefix_v in zip(self.vit_model.blocks, state["prefixes_v"]):
                            block.prefix_modules[task_id].prefix_v.data = prefix_v.to(device)
            except Exception as e:
                print(f"Warning: Could not load PEFT parameters for task {task_id} from {filepath}: {e}")

    def save_lora_parameters(self, filename: str, task_id) -> None:
        if not os.path.exists(filename):
            os.makedirs(filename, exist_ok=True)

        state = {}
        if self.use_prompt:
            state["prompts"] = [block.prompt_modules[task_id].prompt.detach().cpu()
                                 for block in self.vit_model.blocks]

        if self.use_adapter:
            state["adapters"] = [block.adapter_modules[task_id].state_dict()
                                   for block in self.vit_model.blocks]

        if getattr(self, "use_prefix", False):
            state["prefixes_k"] = [block.prefix_modules[task_id].prefix_k.detach().cpu()
                                    for block in self.vit_model.blocks]
            state["prefixes_v"] = [block.prefix_modules[task_id].prefix_v.detach().cpu()
                                    for block in self.vit_model.blocks]

        if len(state) > 0:
            torch.save(state, os.path.join(filename, f"peft_task_{task_id}.pt"))


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x


class LoRA_ViT(nn.Module):
    """Applies low-rank adaptation to a vision transformer.
    Args:
        vit_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.
    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """
    def __init__(self, vit_model: ViT, r: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_ViT, self).__init__()

        assert r > 0
        base_vit_dim = vit_model.transformer.blocks[0].attn.proj_q.in_features
        dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.transformer.blocks)))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.transformer.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_q_linear = blk.attn.proj_q
            w_v_linear = blk.attn.proj_v
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.proj_q = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
            blk.attn.proj_v = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)

        self.reset_parameters()
        self.lora_vit = vit_model
        if num_classes > 0:
            self.lora_vit.fc = nn.Linear(vit_model.fc.in_features, num_classes)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor, sample_task_id) -> Tensor:
        return self.lora_vit(x, sample_task_id)


class _LoRA_qkv_timm(nn.Module):
    """
    In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x)) #* self.scaling_factor
        new_v = self.linear_b_v(self.linear_a_v(x)) #* self.scaling_factor
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv
    
class _LoRA_qkv_timm_train(nn.Module):
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v, #linear_a_q1, linear_b_q1, linear_a_v1, linear_b_v1,
        task_id, saved_A, saved_B, t_layer_i, rank, scaling_factor, scaling_factor_prev, parent_vit_tt_weights, parent_vit_eval_mode, eval1=False):
        super().__init__()
        self.linear_a_q = linear_a_q.cuda()
        self.linear_b_q = linear_b_q.cuda()
        self.linear_a_v = linear_a_v.cuda()
        self.linear_b_v = linear_b_v.cuda()

        self.scaling_factor = scaling_factor.cuda()
        self.scaling_factor_prev = scaling_factor_prev.cuda()

        self.tt_weights = parent_vit_tt_weights
        self.eval_mode = parent_vit_eval_mode

        self.task_id = task_id
        self.qkv = qkv
        self.dim = qkv.in_features
        self.saved_A = saved_A
        self.saved_B = saved_B
        self.t_layer_i = t_layer_i
        self.rank = rank
        self.eval = eval1

        self.w_a_linear_q = nn.Linear(self.dim, self.rank, bias=False)
        self.w_b_linear_q = nn.Linear(self.rank, self.dim, bias=False)
        self.w_a_linear_v = nn.Linear(self.dim, self.rank, bias=False)
        self.w_b_linear_v = nn.Linear(self.rank, self.dim, bias=False)

    def forward(self, x):

        
         
        new_q, new_v = 0, 0

        if self.eval_mode:

            nn.init.zeros_(self.w_a_linear_q.weight)
            nn.init.zeros_(self.w_b_linear_q.weight)
            nn.init.zeros_(self.w_a_linear_v.weight)
            nn.init.zeros_(self.w_b_linear_v.weight)
            
            with torch.no_grad():
                for i in range(self.task_id):
                    #i = self.parent_vit.internal_task_id
                    saved_A_i, saved_B_i = self.saved_A['saved_A_'+str(i)], self.saved_B['saved_B_'+str(i)]
                    Q, V = list(enumerate(zip(saved_A_i,saved_B_i)))[self.t_layer_i*2: self.t_layer_i*2+2]
                    _, (A_q, B_q) = Q
                    _, (A_v, B_v) = V

                    self.w_a_linear_q.weight.data += self.tt_weights[i] * A_q.weight.data
                    self.w_b_linear_q.weight.data += self.tt_weights[i] * B_q.weight.data
                    self.w_a_linear_v.weight.data += self.tt_weights[i] * A_v.weight.data
                    self.w_b_linear_v.weight.data += self.tt_weights[i] * B_v.weight.data

                self.w_a_linear_q.weight.data += self.tt_weights[self.task_id] * self.linear_a_q.weight.data
                self.w_b_linear_q.weight.data += self.tt_weights[self.task_id] * self.linear_b_q.weight.data
                self.w_a_linear_v.weight.data += self.tt_weights[self.task_id] * self.linear_a_v.weight.data
                self.w_b_linear_v.weight.data += self.tt_weights[self.task_id] * self.linear_b_v.weight.data


                new_q =  self.scaling_factor[0](self.w_b_linear_q(self.w_a_linear_q(x)))
                new_v = self.scaling_factor[0](self.w_b_linear_v(self.w_a_linear_v(x)))
        
        else:
            new_q = self.scaling_factor[0](self.linear_b_q(self.linear_a_q(x)))#self.scaling_factor[0]( self.linear_b_q(self.linear_a_q(x)) )
            new_v = self.scaling_factor[0](self.linear_b_v(self.linear_a_v(x)))#self.scaling_factor[0]( self.linear_b_v(self.linear_a_v(x)) )
        qkv = self.qkv(x) 
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv

class _LoRA_qkv_timm_eval(nn.Module):
    """
    In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """
    def __init__(self, task_id, qkv: nn.Module, saved_A, saved_B, t_layer_i, rank, scaling_factor,  scaling_factor_prev, save_file, parent_vit):
        super().__init__()
        self.task_id = task_id
        self.qkv = qkv
        self.dim = qkv.in_features
        self.saved_A = saved_A
        self.saved_B = saved_B
        self.t_layer_i = t_layer_i
        self.rank = rank

        self.save_file = save_file
        self.scaling_factor = scaling_factor.cuda()
        self.scaling_factor_prev = scaling_factor_prev.cuda()

        self.parent_vit = parent_vit

        self.w_a_linear_q = nn.Linear(self.dim, self.rank, bias=False)
        self.w_b_linear_q = nn.Linear(self.rank, self.dim, bias=False)
        self.w_a_linear_v = nn.Linear(self.dim, self.rank, bias=False)
        self.w_b_linear_v = nn.Linear(self.rank, self.dim, bias=False)


    def forward(self, x):
        new_q, new_v = 0, 0

        

        nn.init.zeros_(self.w_a_linear_q.weight)
        nn.init.zeros_(self.w_b_linear_q.weight)
        nn.init.zeros_(self.w_a_linear_v.weight)
        nn.init.zeros_(self.w_b_linear_v.weight)


        file_path = self.save_file+'scaling_factor'+str(self.task_id-1)+'.pt'
        scaling_param = torch.load(file_path)
        if True: # weighted average
            for i in range(self.task_id+1):
                #i = self.parent_vit.internal_task_id
                saved_A_i, saved_B_i = self.saved_A['saved_A_'+str(i)], self.saved_B['saved_B_'+str(i)]
                Q, V = list(enumerate(zip(saved_A_i,saved_B_i)))[self.t_layer_i*2: self.t_layer_i*2+2]
                _, (A_q, B_q) = Q
                _, (A_v, B_v) = V

                self.w_a_linear_q.weight.data += self.parent_vit.tt_weights[i] * A_q.weight.data# (self.scaling_factor_prev[i]**0.5) * self.parent_vit.tt_weights[i] * A_q.weight.data
                self.w_b_linear_q.weight.data += self.parent_vit.tt_weights[i] * B_q.weight.data#(self.scaling_factor_prev[i]**0.5) * self.parent_vit.tt_weights[i] * B_q.weight.data
                self.w_a_linear_v.weight.data += self.parent_vit.tt_weights[i] * A_v.weight.data#(self.scaling_factor_prev[i]**0.5) * self.parent_vit.tt_weights[i] * A_v.weight.data
                self.w_b_linear_v.weight.data += self.parent_vit.tt_weights[i] * B_v.weight.data#(self.scaling_factor_prev[i]**0.5) * self.parent_vit.tt_weights[i] * B_v.weight.data

            new_q =  self.scaling_factor[0](self.w_b_linear_q(self.w_a_linear_q(x)))#/ (torch.norm(w_b_linear_q.weight)* torch.norm(w_a_linear_q.weight) )  )
            new_v = self.scaling_factor[0](self.w_b_linear_v(self.w_a_linear_v(x)))#/ (torch.norm(w_b_linear_v.weight)* torch.norm(w_a_linear_v.weight) )  )
        
        if False: #weighted concat
            for i in range(self.task_id):
                #i = self.parent_vit.internal_task_id
                saved_A_i, saved_B_i = self.saved_A['saved_A_'+str(i)], self.saved_B['saved_B_'+str(i)]
                Q, V = list(enumerate(zip(saved_A_i,saved_B_i)))[self.t_layer_i*2: self.t_layer_i*2+2]
                _, (A_q, B_q) = Q
                _, (A_v, B_v) = V

                self.w_a_linear_q.weight = self.parent_vit.tt_weights[i] * A_q.weight.data# (self.scaling_factor_prev[i]**0.5) * self.parent_vit.tt_weights[i] * A_q.weight.data
                self.w_b_linear_q.weight = self.parent_vit.tt_weights[i] * B_q.weight.data#(self.scaling_factor_prev[i]**0.5) * self.parent_vit.tt_weights[i] * B_q.weight.data
                self.w_a_linear_v.weight = self.parent_vit.tt_weights[i] * A_v.weight.data#(self.scaling_factor_prev[i]**0.5) * self.parent_vit.tt_weights[i] * A_v.weight.data
                self.w_b_linear_v.weight = self.parent_vit.tt_weights[i] * B_v.weight.data#(self.scaling_factor_prev[i]**0.5) * self.parent_vit.tt_weights[i] * B_v.weight.data

                new_q +=  self.w_b_linear_q(self.w_a_linear_q(x))#/ (torch.norm(w_b_linear_q.weight)* torch.norm(w_a_linear_q.weight) )  )
                new_v += self.w_b_linear_v(self.w_a_linear_v(x))#/ (torch.norm(w_b_linear_v.weight)* torch

        #new_q = self.scaling_factor_prev[i]( w_b_linear_q(w_a_linear_q(x)))#/ (torch.norm(w_b_linear_q.weight)* torch.norm(w_a_linear_q.weight) )  )
        #new_v = self.scaling_factor_prev[i]( w_b_linear_v(w_a_linear_v(x)))#/ (torch.norm(w_b_linear_v.weight)* torch.norm(w_a_linear_v.weight) )  )
        

        #new_q += self.scaling_factor[0]( w_b_linear_q(w_a_linear_q(x)) )
        #new_v += self.scaling_factor[0]( w_b_linear_v(w_a_linear_v(x)) )
 
        qkv = self.qkv(x) 
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv
    


class ParameterWrapper(nn.Module):
    def __init__(self, param):
        super(ParameterWrapper, self).__init__()
        self.param = param
    
    def forward(self, x):
        # print('x, param', x.device(), self.pram.device())
        return x * self.param
    
class MyLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyLinear, self).__init__()
        self.linear_b_q = nn.Linear(input_dim, output_dim, bias=False)
        self.linear_b_q = utils.weight_norm(self.linear_b_q)

    def forward(self, x):
        return self.linear_b_q(x)


class LoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int, num_classes: int = 0, increment=10, filepath = './', lora_layer=None, eval=False, index=True, cur_task_index=None):
        super(LoRA_ViT_timm, self).__init__()

        assert r > 0
        self.rank =r
        self.base_vit = copy.deepcopy(vit_model)
        self.internal_task_id = 0
        #self.tt_weights = torch.ones(self.task_id)
        self.lora_alpha = 1.0
        self.multi_rank = [r,r,r,r,r,r,r,r,r,r,r,r]#[4,4,4,4,4,4,4,4,4,4,4,4]#[8,8,8,8,8,8,8,8,8,8,8,8]#[10,10,10,10,10,10,10,10,10,10,10,10]#[4,4,4,4,4,4,4,4,4,4,4,4]# [10,10,10,10,10,10,10,10,10,10,10,10]#[20,20,20,20,20,20,20,20,20,20,20,20]#[5,5,5,5,10,10,10,10,15,15,15,15]
        self.eval_mode = False

        if not eval:
            self.save_file = filepath
            self.increment = increment
            print('save_file', self.save_file)


        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))


        self.w_As, self.w_Bs = [], []  # These are linear layers

        
        if index:
            print('Initialize task-id and curtask id')
            self.task_id, self.cur_id = 0,0
        
        if cur_task_index != None:
            # print('Update the network!!!', cur_task_index)
            self.task_id = cur_task_index

        self.tt_weights = torch.ones(self.task_id)

        # freeze the saved part
        for param in self.base_vit.parameters():
            param.requires_grad = False


        for param in vit_model.parameters():
            param.requires_grad = False

        saved_lora_A, saved_lora_B = {}, {}
        for i in range(self.task_id):
            file_path = self.save_file+'lora_w_a_'+str(i)+'.pt'
            saved_lora_A['saved_A_'+str(i)] = torch.load(file_path)
            file_path = self.save_file+'lora_w_b_'+str(i)+'.pt'
            saved_lora_B['saved_B_'+str(i)] = torch.load(file_path)

        scaling_factor = nn.Parameter(torch.Tensor([self.lora_alpha/self.multi_rank[0]]), requires_grad=False)
        self.wrapped_param = nn.ModuleList([ParameterWrapper(scaling_factor)])
        self.wrapped_param_prev = nn.ModuleList([ParameterWrapper(nn.Parameter(torch.Tensor([self.lora_alpha/self.multi_rank[0]]), requires_grad=False)) for _ in range(20)])

        # Do the surgery 
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, self.multi_rank[t_layer_i], bias=False)
            w_b_linear_q = nn.Linear(self.multi_rank[t_layer_i], self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, self.multi_rank[t_layer_i], bias=False)
            w_b_linear_v = nn.Linear(self.multi_rank[t_layer_i], self.dim, bias=False)

            
            if self.task_id > 0:
            #for i in range(self.task_id):
                i = self.task_id-1
                saved_A_i, saved_B_i = saved_lora_A['saved_A_'+str(i)], saved_lora_B['saved_B_'+str(i)]
                Q, V = list(enumerate(zip(saved_A_i,saved_B_i)))[t_layer_i*2: t_layer_i*2+2]
                _, (A_q, B_q) = Q
                _, (A_v, B_v) = V

                #print('mean', A_q.weight.mean())
                w_a_linear_q.weight = Parameter(A_q.weight.clone())
                w_a_linear_q.weight.requires_grad = True
                #w_a_linear_q.to(x.device)
                w_b_linear_q.weight = Parameter(B_q.weight.clone())
                w_b_linear_q.weight.requires_grad = True
                #w_b_linear_q.to(x.device)
                w_a_linear_v.weight = Parameter(A_v.weight.clone())
                w_a_linear_v.weight.requires_grad = True
                #w_a_linear_v.to(x.device)
                w_b_linear_v.weight = Parameter(B_v.weight.clone())
                w_b_linear_v.weight.requires_grad = True 
                #w_b_linear_v.to(x.device)

            #nn.init.kaiming_uniform_(w_a_linear_q.weight, a=math.sqrt(5))
            #nn.init.kaiming_uniform_(w_a_linear_v.weight, a=math.sqrt(5))
            #nn.init.zeros_(w_b_linear_q.weight)
            #nn.init.zeros_(w_b_linear_v.weight)
            #nn.init.zeros_(w_a_linear_q.weight)
            #nn.init.zeros_(w_a_linear_v.weight)
            #w_a_linear_q.weight.requires_grad = True
            #w_b_linear_q.weight.requires_grad = True
            #w_a_linear_v.weight.requires_grad = True
            #w_b_linear_v.weight.requires_grad = True 

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            if not eval:
                blk.attn.qkv = _LoRA_qkv_timm_train(
                    w_qkv_linear, w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v, 
                    self.task_id, saved_lora_A, saved_lora_B, t_layer_i, self.multi_rank[t_layer_i] , self.wrapped_param, self.wrapped_param_prev, parent_vit_tt_weights=self.tt_weights, parent_vit_eval_mode=self.eval_mode, eval1=False
                )
            else:
                blk.attn.qkv = _LoRA_qkv_timm_eval(self.task_id, w_qkv_linear, saved_lora_A, saved_lora_B, t_layer_i, self.multi_rank[t_layer_i], self.wrapped_param, self.wrapped_param_prev, self.save_file, parent_vit=self) 

        if self.task_id == 0:
            self.reset_parameters()
        self.lora_vit = vit_model
        if not eval:
            self.lora_vit.head = torch.nn.Identity()
        else:
            self.reset_lora_vit_head()

    def set_internal_task_id(self, task_id):
        self.internal_task_id = task_id

    def set_internal_tt_weights(self, weights):
        self.tt_weights = weights

        for t_layer_i, blk in enumerate(self.lora_vit.blocks):
            blk.attn.qkv.tt_weights = self.tt_weights

    # def set_task_internal_tt_weights(self, task_id=0):
    #     self.tt_weights = torch.zeros(self.task_id)
    #     self.tt_weights[self.task_id] = 1

    def reset_internal_tt_weights(self, full_weight_task_id=0):
        self.tt_weights = torch.zeros(self.task_id+1)
        self.tt_weights[full_weight_task_id] = 1

        for t_layer_i, blk in enumerate(self.lora_vit.blocks):
            blk.attn.qkv.tt_weights = self.tt_weights

    def reset_lora_vit_head(self):
        task_incremental = self.increment
        self.lora_vit.head = self.generate_fc(768, (self.task_id)*task_incremental).cuda()
        temp_weights = torch.load(self.save_file+'CLs_weight'+str(self.task_id-1)+'.pt') 
        temp_bias = torch.load(self.save_file+'CLs_bias'+str(self.task_id-1)+'.pt') 

        self.lora_vit.head.weight.data = temp_weights.data.cuda()
        self.lora_vit.head.bias.data = temp_bias.data.cuda()

    def activate_eval(self):
        self.eval_mode = True
        for t_layer_i, blk in enumerate(self.lora_vit.blocks):
            blk.attn.qkv.eval_mode = True
    
    def deactivate_eval(self):
        self.eval_mode = False
        for t_layer_i, blk in enumerate(self.lora_vit.blocks):
            blk.attn.qkv.eval_mode = False

    # This part is only used during the evaluation
    def reset(self, eval=False):
        self.__init__(self.base_vit, self.rank, lora_layer=None, eval=eval, index=False)

    def reset_parameters(self) -> None:
        # if self.task_id ==0: 
            for w_A in self.w_As:
                nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
                # nn.init.kaiming_uniform_(w_A.linear_b_q.weight, a=math.sqrt(5) )
            for w_B in self.w_Bs:
                nn.init.zeros_(w_B.weight)


    def save_wrap_param(self, filename):
        if self.task_id ==1:   
            scaling_param = torch.zeros(20,20)
        else:
            scaling_param = torch.load(filename + 'scaling_factor'+str(self.task_id-2)+'.pt')
        i = self.task_id-1
        # print('save i', i)
        for j in range(i+1):
            if j == i:
                scaling_param[i][j] = self.wrapped_param[0].param.clone()
            else:
                scaling_param[i][j] = self.wrapped_param_prev[j].param.clone()  
        torch.save(scaling_param, filename + 'scaling_factor'+str(self.task_id-1)+'.pt')
        
    def save_lora_parameters(self, filename: str, task_id) -> None:
        self.task_id += 1
        if not os.path.exists(filename):
           os.makedirs(filename)
        torch.save(self.w_As, filename + 'lora_w_a_'+str(task_id)+'.pt')
        torch.save(self.w_Bs, filename + 'lora_w_b_'+str(task_id)+'.pt')

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def load_eval_vit(self):
        self.lora_vit = copy.deepcopy(self.base_vit)
        saved_lora_A, saved_lora_B = {}, {}
        for i in range(self.task_id):
            file_path = self.save_file+'lora_w_a_'+str(i)+'.pt'
            saved_lora_A['saved_A_'+str(i)] = torch.load(file_path)
            file_path = self.save_file+'lora_w_b_'+str(i)+'.pt'
            saved_lora_B['saved_B_'+str(i)] = torch.load(file_path)

        # for param in self.eval_vit.parameters():
        for param in self.lora_vit.parameters():
            param.requires_grad = False
        
        # for t_layer_i, blk in enumerate(self.eval_vit.blocks):
        for t_layer_i, blk in enumerate(self.lora_vit.blocks):
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            blk.attn.qkv = _LoRA_qkv_timm_eval(self.task_id, w_qkv_linear, saved_lora_A, saved_lora_B, t_layer_i, self.multi_rank[t_layer_i])    
        self.reset_lora_vit_head()

    def compute_ortho_loss(self):
        loss = torch.tensor(0).float().cuda()
        # print('task_id', self.task_id)
        for i in range(self.task_id):
            file_path = self.save_file+'lora_w_a_'+str(i)+'.pt'
            if os.path.exists(file_path):
                w_As = torch.load(file_path)
                num_layer = len(self.w_As)
                for j in range(num_layer):
                    temp = torch.matmul(w_As[j].weight.to(self.w_As[j].weight.device), self.w_As[j].weight.t())
                    temp = torch.sum(torch.square(temp))
                    loss = loss.to(self.w_As[j].weight.device)
                    loss += temp
        return loss
    
    def forward(self, x: Tensor, loss= False, eval=False) -> Tensor:
        if eval:
            self.reset(eval=True)
            return self.lora_vit(x)
        elif loss:
            loss = self.compute_ortho_loss()
            return self.lora_vit(x), loss
        else:
            return self.lora_vit(x)
        


