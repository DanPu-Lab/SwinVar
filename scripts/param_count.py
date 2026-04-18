from typing import Dict, List, Set, Tuple
import torch
import torch.nn as nn

class ActivatedParamCounter:
    """
    用法：
        counter = ActivatedParamCounterForYourMoE(model)
        with torch.no_grad():
            _ = model(x)     # 正常前向一次
        active = counter.activated_params_once(trainable_only=False)
        counter.clear()     # 若要复用统计缓存
        counter.remove()    # 用完卸载 hooks
    """
    def __init__(self, model: nn.Module):
        self.model = model
        # 找出所有 MoE 模块，并建立 gate -> moe 的映射
        self.moes: List[nn.Module] = []
        self.gate_to_moe: Dict[nn.Module, nn.Module] = {}
        self._discover_moes_and_gates()

        # 每个 MoE 在“本次前向”中被使用的 expert id（去重后）
        self.moe_used_experts: Dict[nn.Module, Set[int]] = {m: set() for m in self.moes}

        # 注册 gate 的 forward hook，捕获 (weights, indices) 的 indices
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        for gate, moe in self.gate_to_moe.items():
            h = gate.register_forward_hook(self._gate_hook(moe))
            self.hooks.append(h)

        # 预先缓存：所有 MoE 的参数集合（避免重复计数）
        self._all_moe_param_ids: Set[int] = set()
        for moe in self.moes:
            for p in moe.parameters():
                self._all_moe_param_ids.add(id(p))

        # 预计算每个模块内各部分参数量（加速 later）
        self._per_moe_cache: Dict[nn.Module, dict] = {}
        for moe in self.moes:
            self._per_moe_cache[moe] = self._build_moe_param_cache(moe)

    # ---------- 发现 MoE/gate ----------
    def _is_your_moe(self, m: nn.Module) -> bool:
        # 你的 MoE 具有 gate / experts(ModuleList) / shared_experts
        return (
            hasattr(m, "gate")
            and hasattr(m, "experts")
            and isinstance(getattr(m, "experts"), nn.ModuleList)
            and hasattr(m, "shared_experts")
        )

    def _discover_moes_and_gates(self):
        for m in self.model.modules():
            if self._is_your_moe(m):
                self.moes.append(m)
                gate = getattr(m, "gate")
                self.gate_to_moe[gate] = m

    # ---------- Hook：抓取 Gate.forward 输出的 indices ----------
    def _gate_hook(self, moe_module: nn.Module):
        def fn(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output):
            """
            Gate.forward(x) -> (weights, indices)
            我们只取 indices（LongTensor），形状 [N, topk]。
            """
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                indices = output[1]
            else:
                indices = None
            if torch.is_tensor(indices) and indices.dtype in (torch.int32, torch.int64):
                # 累积被使用的 expert id（去重）
                used = indices.detach().unique().tolist()
                self.moe_used_experts[moe_module].update(int(u) for u in used)
        return fn

    # ---------- 构建 MoE 内部参数的缓存（gate/shared/experts） ----------
    def _sum_params(self, module: nn.Module, trainable_only: bool) -> int:
        if trainable_only:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return sum(p.numel() for p in module.parameters())

    def _build_moe_param_cache(self, moe: nn.Module) -> dict:
        """
        返回：
            {
                "gate_all": int, "gate_trainable": int,
                "shared_all": int, "shared_trainable": int,
                "experts_all": List[int], "experts_trainable": List[int],
            }
        """
        gate = getattr(moe, "gate")
        shared = getattr(moe, "shared_experts")
        experts: nn.ModuleList = getattr(moe, "experts")

        cache = {
            "gate_all": self._sum_params(gate, trainable_only=False),
            "gate_trainable": self._sum_params(gate, trainable_only=True),
            "shared_all": self._sum_params(shared, trainable_only=False),
            "shared_trainable": self._sum_params(shared, trainable_only=True),
            "experts_all": [],
            "experts_trainable": [],
        }
        for e in experts:
            cache["experts_all"].append(self._sum_params(e, trainable_only=False))
            cache["experts_trainable"].append(self._sum_params(e, trainable_only=True))
        return cache

    # ---------- 计算一次前向的激活参数 ----------
    def activated_params_once(self, trainable_only: bool = False) -> int:
        """
        计算“本次前向”的激活参数量：
          非MoE的所有参数
        + 每个MoE的 gate + shared_experts
        + 每个MoE本次被选中的 experts
        """
        # (A) 非 MoE 参数
        if trainable_only:
            non_moe = sum(
                p.numel()
                for p in self.model.parameters()
                if id(p) not in self._all_moe_param_ids and p.requires_grad
            )
        else:
            non_moe = sum(
                p.numel()
                for p in self.model.parameters()
                if id(p) not in self._all_moe_param_ids
            )
        total = non_moe

        # (B) 每个 MoE：gate + shared 始终激活；experts 只计入被选中的
        for moe in self.moes:
            cache = self._per_moe_cache[moe]
            if trainable_only:
                total += cache["gate_trainable"] + cache["shared_trainable"]
                for eid in sorted(self.moe_used_experts[moe]):
                    if 0 <= eid < len(cache["experts_trainable"]):
                        total += cache["experts_trainable"][eid]
            else:
                total += cache["gate_all"] + cache["shared_all"]
                for eid in sorted(self.moe_used_experts[moe]):
                    if 0 <= eid < len(cache["experts_all"]):
                        total += cache["experts_all"][eid]
        return total

    # ---------- 其它 ----------
    def clear(self):
        """清空一次前向收集到的 expert 使用记录（可复用该对象继续统计）。"""
        for k in self.moe_used_experts.keys():
            self.moe_used_experts[k].clear()

    def remove(self):
        """卸载所有 hooks。"""
        for h in self.hooks:
            h.remove()
        self.hooks = []

if __name__ == "__main__":
    from swinvar.models.swin_var import SwinVar
    from swinvar.preprocess.parameters import (
        flank_size,
        windows_size,
        CHANNEL_SIZE,
        VARIANT_SIZE,
    )
    args = {
        "input_path": "args.train_input",
        "output_path": "args.train_output",
        "reference": "args.reference",
        "ref_var_ratio": "args.ref_var_ratio",
        "file": f"balance_{2}",
        "epochs": 300,
        "batch_size": 600,
        "feature_size": (windows_size, CHANNEL_SIZE),
        "num_classes": VARIANT_SIZE,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "embed_dim": 192,
        "patch_size": 2,
        "window_size": 3,
        "n_routed_experts": 8,
        "n_activated_experts": 2,
        "n_expert_groups": 1,
        "n_limited_groups": 1,
        "score_func": "sigmoid",
        "route_scale": 1,
        "moe_inter_dim": 64,
        "n_shared_experts": 1,
        "drop_rate": 0.1,
        "drop_path_rate": 0.1,
        "attn_drop_rate": 0.1,
        "lr": 0.001,
        "weight_decay": 0.01,
        "num_workers": "args.num_workers",
        "patience": "args.patience",
        "model_save_path": "best_model.pth",
        "log_file_train": "train_log.txt",
        "log_file_call": "call_log.txt",
        "matplot_save_path": "train_process.png",
        "hyperparams_log": "hyperparams.xlsx",
        "checkpoint": "args.checkpoint",
        "call_batch_size": 5000,
        "call_input_path": "args.call_input",
        "call_file": "pileup",
        "ft": "args.fune_turning",
        "ft_file": "pileup",
        "output_vcf": "args.output_vcf",
        "ft_batch_size": 600,
        "ft_epochs": 50,
        "ft_lr": 0.01,
        "ft_patience": 8,
        "pct_start": 0.3,
        "factor": [1, 1],
    }

    
    model = SwinVar(
        feature_size=args["feature_size"],
        num_classes=args["num_classes"],
        embed_dim=args["embed_dim"],
        patch_size=args["patch_size"],
        window_size=args["window_size"],
        n_routed_experts=args["n_routed_experts"],
        n_activated_experts=args["n_activated_experts"],
        n_expert_groups=args["n_expert_groups"],
        n_limited_groups=args["n_limited_groups"],
        score_func=args["score_func"],
        route_scale=args["route_scale"],
        moe_inter_dim=args["moe_inter_dim"],
        n_shared_experts=args["n_shared_experts"],
        depths=args["depths"],
        num_heads=args["num_heads"],
        drop_rate=args["drop_rate"],
        drop_path_rate=args["drop_path_rate"],
        attn_drop_rate=args["attn_drop_rate"],
    )
    counter = ActivatedParamCounter(model)\

    x = torch.randn(1, 43, 3, 18)  # 或你的真实输入
    with torch.no_grad():
        _ = model(x)  # 本次前向中，hooks 会记录各 MoE 的 indices

    print("Activated params (all):", counter.activated_params_once(trainable_only=False))
    print("Activated params (trainable):", counter.activated_params_once(trainable_only=True))
    counter.remove()