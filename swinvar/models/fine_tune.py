import math
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Iterable, Union

# ---------------- LoRA 模块：只包 Linear，不改前向接口 ----------------
class LinearLoRA(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 4, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 关键：跟随原层 device/dtype
        device = base_linear.weight.device
        dtype  = base_linear.weight.dtype

        if r > 0:
            self.lora_A = nn.Linear(self.base.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, self.base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            self.lora_A.to(device=device, dtype=dtype)
            self.lora_B.to(device=device, dtype=dtype)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base(x)
        if self.r > 0:
            out = out + self.lora_B(self.lora_A(self.drop(x))) * self.scaling
        return out

# ---------------- 基础遍历工具 ----------------
def iter_blocks(model):
    for li, layer in enumerate(getattr(model, "layers", [])):
        for bi, blk in enumerate(getattr(layer, "blocks", [])):
            yield li, bi, blk

def get_last_k_blocks(model, k: int):
    all_blocks = [(li, bi, blk) for li, bi, blk in iter_blocks(model)]
    return all_blocks[-k:] if k > 0 else []

# ---------------- 解冻策略 ----------------
def _freeze_all(model):
    for _, p in model.named_parameters():
        p.requires_grad = False

def _unfreeze_heads_and_global_norm(model):
    for head in [getattr(model, "head1", None), getattr(model, "head2", None), getattr(model, "head3", None)]:
        if head is not None:
            for p in head.parameters(): p.requires_grad = True
    if hasattr(model, "norm2"):
        for p in model.norm2.parameters(): p.requires_grad = True

def unfreeze_last_stage_core(model):
    """解冻最后一段：Norm + 注意力(qkv/proj) + MoE(gate/experts/shared_experts) + heads + norm2"""
    _freeze_all(model)
    _unfreeze_heads_and_global_norm(model)
    last_layer = model.layers[-1]
    for blk in last_layer.blocks:
        # Norm
        for m in [blk.norm1, blk.norm2]:
            for p in m.parameters(): p.requires_grad = True
        # Attention
        for name in ["qkv", "proj"]:
            mod = getattr(blk.attn, name, None)
            if isinstance(mod, nn.Linear):
                for p in mod.parameters(): p.requires_grad = True
            # 若已被 LoRA 替换为 LinearLoRA，也有 parameters()
            elif isinstance(mod, LinearLoRA):
                for p in mod.parameters(): p.requires_grad = True
        # MoE
        for p in blk.moe.gate.parameters(): p.requires_grad = True
        for expert in blk.moe.experts:
            for p in expert.parameters(): p.requires_grad = True
        for p in blk.moe.shared_experts.parameters(): p.requires_grad = True

def inject_lora_for_attention(model, last_k_blocks=4, r=4, alpha=16, dropout=0.0):
    targets = get_last_k_blocks(model, last_k_blocks)
    replaced = []
    for li, bi, blk in targets:
        for name in ["qkv", "proj"]:
            lin = getattr(blk.attn, name, None)
            if isinstance(lin, nn.Linear):
                lora = LinearLoRA(lin, r=r, alpha=alpha, dropout=dropout)
                # 双保险：再按原层 device/dtype 迁移一次
                lora.to(device=lin.weight.device, dtype=lin.weight.dtype)
                setattr(blk.attn, name, lora)
                replaced.append(f"layers.{li}.blocks.{bi}.attn.{name}")
    return replaced

def unfreeze_norm_and_bias(model):
    for _, m in model.named_modules():
        if isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
            for p in m.parameters(): p.requires_grad = True
    for n, p in model.named_parameters():
        if n.endswith(".bias"):
            p.requires_grad = True

# ---------------- 分层学习率（LLRD）参数组 ----------------
def build_param_groups_with_llrd(
    model,
    base_lr: float,
    head_lr: float,
    weight_decay: float = 0.05,
    layer_decay: float = 0.75
):
    groups: Dict[str, Dict] = {}

    def add_param(p: nn.Parameter, lr: float, wd: float, tag: str):
        if not p.requires_grad: return
        key = f"{tag}|lr={lr:.2e}|wd={wd:.2e}"
        if key not in groups:
            groups[key] = {"params": [], "lr": lr, "weight_decay": wd}
        groups[key]["params"].append(p)

    # heads
    for head in [getattr(model, "head1", None), getattr(model, "head2", None), getattr(model, "head3", None)]:
        if head is not None:
            for p in head.parameters():
                add_param(p, head_lr, weight_decay, "heads")

    # global norm2（norm不加wd）
    if hasattr(model, "norm2"):
        for p in model.norm2.parameters():
            add_param(p, base_lr, 0.0, "global_norm")

    layers = list(getattr(model, "layers", []))
    total_stages = len(layers)

    for li, layer in enumerate(layers):
        # 越靠后层学习率越大
        stage_scale = (layer_decay ** (total_stages - 1 - li))
        stage_lr = base_lr * (1.0 / stage_scale)

        for bi, blk in enumerate(layer.blocks):
            # Norm
            for m in [blk.norm1, blk.norm2]:
                for p in m.parameters():
                    add_param(p, stage_lr, 0.0, f"norm_l{li}_b{bi}")
            # Attention（Linear 或 LinearLoRA）
            for name in ["qkv", "proj"]:
                mod = getattr(blk.attn, name, None)
                if mod is None: continue
                for p in mod.parameters():
                    wd = 0.0 if p.ndim == 1 else weight_decay
                    add_param(p, stage_lr, wd, f"attn_{name}_l{li}_b{bi}")
            # MoE
            for p in blk.moe.gate.parameters():
                add_param(p, stage_lr, weight_decay, f"moe_gate_l{li}_b{bi}")
            for expert in blk.moe.experts:
                for p in expert.parameters():
                    add_param(p, stage_lr, weight_decay, f"moe_expert_l{li}_b{bi}")
            for p in blk.moe.shared_experts.parameters():
                add_param(p, stage_lr, weight_decay, f"moe_shared_l{li}")

    # PatchEmbed 若你想训练其 norm，可在外部放开 requires_grad 后自动被纳入
    for n, p in model.patch_embed.named_parameters():
        if p.requires_grad:
            wd = 0.0 if ("norm" in n) else weight_decay
            add_param(p, base_lr, wd, "patch_embed")

    return list(groups.values())

# ---------------- 统一入口：返回 (param_or_groups, used_strategy) ----------------
def prepare_finetune_params(
    model: nn.Module,
    strategy: str = "heads_only",
    last_k_blocks: int = 4,
    lora_r: int = 4,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    base_lr: float = 5e-5,
    head_lr: float = 5e-4,
    weight_decay: float = 0.05,
    layer_decay: float = 0.75,
    use_llrd: bool = True
) -> Tuple[Union[Iterable[nn.Parameter], List[Dict]], str]:
    """
    返回值：
      - 若 use_llrd=True：按层参数组(List[Dict])，可直接传给 AdamW；
      - 若 use_llrd=False：返回 filter(lambda p: p.requires_grad, model.parameters())。
    """
    if strategy not in {"heads_only", "last_stage_core", "last_k_blocks_lora"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    # 先全冻结
    _freeze_all(model)

    if strategy == "heads_only":
        # 只开三个 head（与你当前做法等价）
        for head in [getattr(model, "head1", None), getattr(model, "head2", None), getattr(model, "head3", None)]:
            if head is not None:
                for p in head.parameters(): p.requires_grad = True
        # 是否也开 global norm2：对校准与阈值更友好
        if hasattr(model, "norm2"):
            for p in model.norm2.parameters(): p.requires_grad = True

    elif strategy == "last_stage_core":
        # 解冻最后一段的核心模块（推荐先试）
        unfreeze_last_stage_core(model)

    elif strategy == "last_k_blocks_lora":
        # 在最后K个块的注意力层插 LoRA，并放开 Norm/bias 与 heads（极省参数/时延）
        inject_lora_for_attention(model, last_k_blocks=last_k_blocks, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        unfreeze_norm_and_bias(model)
        for head in [getattr(model, "head1", None), getattr(model, "head2", None), getattr(model, "head3", None)]:
            if head is not None:
                for p in head.parameters(): p.requires_grad = True
        if hasattr(model, "norm2"):
            for p in model.norm2.parameters(): p.requires_grad = True

    if use_llrd:
        # 返回参数组（每组自带 lr/wd），供 AdamW 直接使用
        return build_param_groups_with_llrd(
            model=model,
            base_lr=base_lr,
            head_lr=head_lr,
            weight_decay=weight_decay,
            layer_decay=layer_decay
        ), strategy
    else:
        # 走你原来的 filter + 单一 lr
        trainable = filter(lambda p: p.requires_grad, model.parameters())
        return trainable, strategy

# === finetune_config.py 追加：LoRA 评测期注入/合并工具 ===

def _get_parent_and_name(model: nn.Module, module_path: str):
    """给 'layers.3.blocks.1.attn.qkv' 返回 (parent_module, 'qkv')，便于替换"""
    parts = module_path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

@torch.no_grad()
def merge_all_lora_(model: nn.Module):
    """
    将所有 LinearLoRA 合并到 base Linear：
    W <- W + (B @ A) * (alpha/r)；然后把模块替换回 nn.Linear。
    """
    replace_list = []
    for name, m in model.named_modules():
        if isinstance(m, LinearLoRA) and m.r > 0:
            # 计算 DeltaW
            delta = m.lora_B.weight @ m.lora_A.weight
            delta = delta.to(m.base.weight.dtype) * m.scaling
            m.base.weight.add_(delta)
            # 替换
            parent, leaf = _get_parent_and_name(model, name)
            setattr(parent, leaf, m.base)  # 用合并后的线性层顶替 LoRA 容器
            replace_list.append(name)
    return replace_list

def prepare_lora_for_eval(
    model: nn.Module,
    enable: bool = False,
    last_k_blocks: int = 4,
    r: int = 4,
    alpha: int = 16,
    dropout: float = 0.0,
    lora_path: str = None,
    model_device: str = "cpu",
    merge_after_load: bool = True,
    verbose: bool = False,
):
    """
    评测期 LoRA 注入流程：
      1) 若 enable=True，在最后K个块的 attn.qkv/proj 注入 LoRA 容器；
      2) 之后再加载权重（底座/LoRA都行，strict=False）；
      3) 可选：合并 LoRA 到 base，推理零额外算子。
    返回：None（就地修改 model）
    """
    if not enable:
        return

    # 1) 注入结构（必须在加载权重之前做，以便 state_dict 有对应 key）
    inject_lora_for_attention(model, last_k_blocks=last_k_blocks, r=r, alpha=alpha, dropout=dropout)

    # 2) 如有单独 LoRA 权重，稍后在外部 load_state_dict(strict=False) 再额外加载；这里仅提供便捷接口
    if lora_path:
        sd_lora = torch.load(lora_path, map_location=model_device)
        missing, unexpected = model.load_state_dict(sd_lora, strict=False)
        if verbose:
            print(f"[LoRA] loaded from {lora_path}. missing={len(missing)}, unexpected={len(unexpected)}")

    # 3) 可选合并（推荐 True）
    if merge_after_load:
        replaced = merge_all_lora_(model)
        if verbose:
            print(f"[LoRA] merged into base linear for {len(replaced)} modules.")
