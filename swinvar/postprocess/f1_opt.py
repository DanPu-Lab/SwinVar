# swinvar/postprocess/f1_opt.py
import math
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, List, Dict, Any

# ---------- 1) 温度缩放（Guo et al., ICML'17） ----------
class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_t) + 1e-6
        return logits / T


def fit_temperature(logits: torch.Tensor,
                    labels: torch.Tensor,
                    max_iter: int = 200,
                    lr: float = 0.05) -> TempScaler:
    """
    注意：不要在 no_grad() 里调用本函数
    logits: (N, K) ; labels: (N,) Long
    """
    # 我们不需要 logits 对模型权重的梯度，但需要对 log_t 的梯度
    logits = logits.detach()                # OK：仍能对 log_t 回传梯度
    labels = labels.detach().long()

    scaler = TempScaler().to(logits.device)
    scaler.train()                          # 以防万一
    nll = nn.CrossEntropyLoss()
    opt = optim.Adam([scaler.log_t], lr=lr)

    for _ in range(max_iter):
        opt.zero_grad(set_to_none=True)
        # 关键：确保在 enable_grad() 环境中
        with torch.enable_grad():
            loss = nll(scaler(logits), labels)
            loss.backward()
        opt.step()

    return scaler

# ---------- 2) 先验矫正（Saerens et al., 2002） ----------
def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def _prior_adjust(p_tr: np.ndarray, train_priors: np.ndarray, test_priors: np.ndarray, eps=1e-12) -> np.ndarray:
    tpi = np.clip(train_priors.astype(float), eps, 1.0)
    tei = np.clip(test_priors.astype(float),  eps, 1.0)
    r = tei / tpi
    p_adj = p_tr * r[None, :]
    p_adj = p_adj / p_adj.sum(1, keepdims=True)
    return p_adj

# ---------- 3) 验证集上搜索“最大F1”的阈值 ----------
def _search_tau_max_f1(scores_pos: np.ndarray, y_pos: np.ndarray) -> Tuple[float, Dict[str, float]]:
    order = np.argsort(scores_pos)[::-1]
    P = y_pos.sum()
    tp = 0; fp = 0
    best = {"tau":1.0, "precision":0.0, "recall":0.0, "f1":0.0}
    for idx in order:
        if y_pos[idx]: tp += 1
        else:          fp += 1
        prec = tp / max(tp+fp, 1)
        rec  = tp / max(P, 1)
        f1   = (2*prec*rec)/max(prec+rec, 1e-12)
        tau  = scores_pos[idx]
        if f1 > best["f1"]:
            best = {"tau": float(tau), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
    return best["tau"], best

# ---------- 4) 主类：Genotype 的 F1 优化门 ----------
class GenotypeF1Optimizer:
    """
    步骤：
      fit():  用验证集  (logits, labels) 估计 温度缩放+先验矫正 后的 p(non00)，搜F1最优阈值（全局/分箱）
      predict(): 在测试/部署时应用，返回 (pred_non00, final_geno)；把低置信“非0/0”打回“0/0”
    """
    def __init__(self, train_priors: List[float], test_priors: List[float], geno_index_00: int = 0):
        self.train_priors = np.asarray(train_priors, float)
        self.test_priors  = np.asarray(test_priors,  float)
        self.geno_index_00 = geno_index_00
        self.temp: Optional[TempScaler] = None
        self.tau_global: float = 0.5
        self.bin_edges: Optional[np.ndarray] = None
        self.bin_taus: Optional[np.ndarray] = None

    def fit(self,
            val_logits: torch.Tensor,   # (Nv,4)
            val_labels: torch.Tensor,   # (Nv,)
            *,
            bin_feature: Optional[np.ndarray] = None,  # (Nv,) 例如覆盖度/映射质量/难区分箱
            n_bins: int = 0):
        # 1) 温度缩放
        self.temp = fit_temperature(val_logits, val_labels)
        with torch.no_grad():
            cal_logits = self.temp(val_logits).cpu()
        p_tr = _softmax_np(cal_logits.numpy())

        # 2) 先验矫正到部署先验
        p_adj = _prior_adjust(p_tr, self.train_priors, self.test_priors)
        p_non00 = 1.0 - p_adj[:, self.geno_index_00]
        y_pos   = (val_labels.cpu().numpy() != self.geno_index_00)

        # 3) 全局阈值（最大F1）
        tau, _ = _search_tau_max_f1(p_non00, y_pos)
        self.tau_global = float(tau)

        # 4) 可选：分箱动态阈值（困难区域用更保守阈值）
        if bin_feature is not None and n_bins >= 2:
            x = np.asarray(bin_feature, float)
            qs = np.linspace(0, 1, n_bins+1)
            self.bin_edges = np.quantile(x, qs)
            self.bin_edges[0]  = -np.inf
            self.bin_edges[-1] =  np.inf
            taus = []
            for i in range(n_bins):
                m = (x >= self.bin_edges[i]) & (x < self.bin_edges[i+1])
                if m.sum() < 5:
                    taus.append(self.tau_global)
                else:
                    tau_i, _ = _search_tau_max_f1(p_non00[m], y_pos[m])
                    taus.append(float(tau_i))
            self.bin_taus = np.asarray(taus, float)

    @torch.no_grad()
    def _posteriors(self, logits: torch.Tensor) -> np.ndarray:
        cal_logits = self.temp(logits).cpu()
        p_tr = _softmax_np(cal_logits.numpy())
        p_adj = _prior_adjust(p_tr, self.train_priors, self.test_priors)
        return p_adj

    @torch.no_grad()
    def predict(self,
                test_logits: torch.Tensor,    # (Nt,4)
                *,
                bin_feature: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        p_adj = self._posteriors(test_logits)
        p_non00 = 1.0 - p_adj[:, self.geno_index_00]

        if self.bin_edges is None:
            pred_non00 = (p_non00 >= self.tau_global)
        else:
            assert bin_feature is not None
            x = np.asarray(bin_feature, float)
            pred_non00 = np.zeros_like(p_non00, dtype=bool)
            for i in range(len(self.bin_edges)-1):
                m = (x >= self.bin_edges[i]) & (x < self.bin_edges[i+1])
                pred_non00[m] = (p_non00[m] >= self.bin_taus[i])

        final_geno = p_adj.argmax(1)
        final_geno[~pred_non00] = self.geno_index_00
        return pred_non00, final_geno

    # 新增：把需要持久化的内容打包成 dict（Tensor 先转到 CPU）
    def state_dict(self) -> dict:
        temp_log_t = torch.zeros(1)
        if self.temp is not None and isinstance(self.temp, TempScaler):
            temp_log_t = self.temp.log_t.detach().cpu().clone()
        return {
            "train_priors": self.train_priors.astype(float),
            "test_priors":  self.test_priors.astype(float),
            "geno_index_00": int(self.geno_index_00),
            "tau_global":    float(self.tau_global),
            "temp_log_t":    temp_log_t,
        }

    # 新增：从 dict 还原（log_t 拷回 Param；设备稍后再 .to(device)）
    def load_state_dict(self, sd: dict):
        self.train_priors = np.asarray(sd["train_priors"], float)
        self.test_priors  = np.asarray(sd["test_priors"],  float)
        self.geno_index_00 = int(sd["geno_index_00"])
        self.tau_global    = float(sd["tau_global"])
        self.temp = TempScaler()
        with torch.no_grad():
            self.temp.log_t.copy_(sd.get("temp_log_t", torch.zeros(1)))
        return self

    # ---------- 持久化 ----------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "train_priors": self.train_priors.tolist(),
            "test_priors":  self.test_priors.tolist(),
            "geno_index_00": self.geno_index_00,
            "tau_global": self.tau_global,
            "bin_edges": None if self.bin_edges is None else self.bin_edges.tolist(),
            "bin_taus":  None if self.bin_taus  is None else self.bin_taus.tolist(),
            "temp_log_t": float(self.temp.log_t.detach().cpu().item()) if self.temp is not None else 0.0,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> "GenotypeF1Optimizer":
        state = torch.load(path, map_location="cpu")
        obj = cls(state["train_priors"], state["test_priors"], state["geno_index_00"])
        obj.tau_global = state["tau_global"]
        obj.bin_edges  = None if state["bin_edges"] is None else np.asarray(state["bin_edges"], float)
        obj.bin_taus   = None if state["bin_taus"]  is None else np.asarray(state["bin_taus"],  float)
        obj.temp = TempScaler()
        with torch.no_grad():
            obj.temp.log_t.copy_(torch.tensor([state["temp_log_t"]], dtype=torch.float32))
        return obj
