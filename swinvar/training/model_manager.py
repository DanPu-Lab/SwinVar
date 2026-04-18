import os
import torch
from torch_optimizer.lookahead import Lookahead
from swinvar.models.swin_var import SwinVar
from swinvar.models.fine_tune import prepare_finetune_params
from swinvar.models.focal_loss import MultiTaskLoss


class ModelManager:
    """模型管理类"""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = None
        
    def create_model(self):
        """创建模型"""
        self.model = SwinVar(
            feature_size=self.args["feature_size"],
            num_classes=self.args["num_classes"],
            embed_dim=self.args["embed_dim"],
            patch_size=self.args["patch_size"],
            window_size=self.args["window_size"],
            n_routed_experts=self.args["n_routed_experts"],
            n_activated_experts=self.args["n_activated_experts"],
            n_expert_groups=self.args["n_expert_groups"],
            n_limited_groups=self.args["n_limited_groups"],
            score_func=self.args["score_func"],
            route_scale=self.args["route_scale"],
            moe_inter_dim=self.args["moe_inter_dim"],
            n_shared_experts=self.args["n_shared_experts"],
            depths=self.args["depths"],
            num_heads=self.args["num_heads"],
            drop_rate=self.args["drop_rate"],
            drop_path_rate=self.args["drop_path_rate"],
            attn_drop_rate=self.args["attn_drop_rate"],
        ).to(self.device)
        
        # 打印模型参数信息
        self._print_model_info()

        return self.model
    
    def _print_model_info(self):
        """打印模型信息"""
        ft_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        if self.args["ft"]:
            print(f"模型的总参数: {total_params}, 微调模型参数: {ft_params}")
        else:
            print(f"模型的总参数: {total_params}")
    
    def setup_finetuning(self, model_save_path):
        """设置微调"""
        self.model.load_state_dict(torch.load(model_save_path, map_location=self.device))
        prepare_finetune_params(self.model)
        self.model.to(self.device)
        # 这些超参可放在 args 里覆盖；给出默认值与你现有命名一致
        param_or_groups, used = prepare_finetune_params(
            model=self.model,
            strategy=self.args["ft_strategy"],
            last_k_blocks=self.args["ft_last_k_blocks"],
            lora_r=self.args["ft_lora_r"],
            lora_alpha=self.args["ft_lora_alpha"],
            lora_dropout=self.args["ft_lora_dropout"],
            base_lr=self.args["ft_base_lr"],
            head_lr=self.args["ft_head_lr"],
            weight_decay=self.args["weight_decay"],
            layer_decay=self.args["ft_layer_decay"],
            use_llrd=self.args["ft_use_llrd"],
        )
        return param_or_groups

    
    def create_optimizer(self, class_weights, param_or_groups):
        """创建优化器"""

        if self.args["ft"]:
            if self.args["ft_use_llrd"]:
                base_optimizer = torch.optim.AdamW(
                    param_or_groups,
                    lr=self.args["ft_base_lr"],
                    weight_decay=self.args["weight_decay"],
                )
            else:
                # 兼容你原来的写法：filter + 单一lr
                params = filter(lambda p: p.requires_grad, self.model.parameters())
                base_optimizer = torch.optim.AdamW(
                    params,
                    lr=self.args["ft_base_lr"],
                    weight_decay=self.args["weight_decay"],
                )
        else:
            # 不做 fine-tune：与原始一致
            params = self.model.parameters()
            base_optimizer = torch.optim.AdamW(
                params,
                lr=self.args["lr"],
                weight_decay=self.args["weight_decay"],
            )

        # 创建Lookahead优化器
        self.optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
        
        # 创建损失函数
        self.criterion = MultiTaskLoss(
            alphas=class_weights['alphas'], 
            gammas=class_weights['gammas'], 
            label_smoothing=0.1
        )

        # 创建混合精度训练的scaler
        self.scaler = torch.amp.GradScaler("cuda")

        return self.optimizer, self.criterion, self.scaler
    
    def save_model(self, save_path):
        """保存模型"""
        torch.save(self.model.state_dict(), save_path)
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint_dict = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
        
        self.model.load_state_dict(checkpoint_dict["model_state_dict"])
        self.optimizer.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        
        # 恢复优化器参数
        for param_group, saved_group in zip(
            self.optimizer.optimizer.param_groups, 
            checkpoint_dict["optimizer_state_dict"]["param_groups"]
        ):
            param_group["lr"] = saved_group["lr"]
            param_group['weight_decay'] = saved_group['weight_decay']
        
        return checkpoint_dict
    
    def save_checkpoint(self, checkpoint_dict, checkpoint_path):
        """保存检查点"""
        checkpoint_dict.update({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.optimizer.state_dict(),
        })
        torch.save(checkpoint_dict, checkpoint_path)
    
    def get_model_params_count(self):
        """获取模型参数数量"""
        ft_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params, ft_params
    
    def train(self):
        """设置模型为训练模式"""
        self.model.train()
    
    def eval(self):
        """设置模型为评估模式"""
        self.model.eval()
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)
    
    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()
    
    def step(self):
        """优化器步进"""
        self.scaler.step(self.optimizer)
    
    def update_scaler(self):
        """更新scaler"""
        self.scaler.update()
    
    def scale_loss(self, loss):
        """缩放损失"""
        return self.scaler.scale(loss)
    
    def backward(self, scaled_loss):
        """反向传播"""
        scaled_loss.backward()