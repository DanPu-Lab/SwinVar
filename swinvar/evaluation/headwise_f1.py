# -*- coding: utf-8 -*-
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ---------- 小工具 ----------
def _auto_avg(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """二分类用 'binary'，多分类用 'macro'（也可改 'weighted'）"""
    labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    return "binary" if len(labels) <= 2 else "macro"

def _safe_f1(y_true: np.ndarray, y_pred: np.ndarray, average: Optional[str]=None) -> float:
    if y_true.size == 0:
        return float("nan")
    labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    avg = average or ("binary" if len(labels) == 2 else "macro")
    kwargs = {}
    if avg == "binary":
        # 确保 pos_label 一定在 labels 里（例如子集标签是 [4,5]）
        kwargs["pos_label"] = labels.max()
    return f1_score(y_true, y_pred, average=avg, zero_division=0, **kwargs)


def _cls_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    if y_true.size == 0:
        return {"empty": True}
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)

def _cm(y_true: np.ndarray, y_pred: np.ndarray) -> List[List[int]]:
    if y_true.size == 0:
        return []
    cm = confusion_matrix(y_true, y_pred).tolist()
    return cm

# ---------- 主函数 ----------
def evaluate_heads(
    results: Dict[str, List[int]],
    *,
    # genotype 类别索引定义（默认：0->0/0, 1->1/0, 2->1/1, 3->1/2）
    geno_index_00: int = 0,
    geno_index_het: int = 1,
    geno_index_hom: int = 2,
    geno_index_het_alt: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    输入：
        results: 你评测循环里填充的 self.results 字典（numpy/int 列表皆可）
                 需要键：
                   - 'genotype_predictions', 'genotype_labels'
                   - 'variant_1_predictions', 'variant_1_labels'
                   - 'variant_2_predictions', 'variant_2_labels'
    输出：
        dict，包含：每个头的 F1（全局/条件）、classification_report、confusion_matrix、
        以及“因 genotype 误触发/漏触发”的样本统计。
    """
    V6_LABELS = np.array([0, 1, 2, 3, 4, 5])  # A,C,G,T,Insert,Deletion 的整数编码
    
    # 转成 numpy
    g_pred = np.asarray(results["genotype_predictions"])
    g_true = np.asarray(results["genotype_labels"])
    v1_pred = np.asarray(results["variant_1_predictions"])
    v1_true = np.asarray(results["variant_1_labels"])
    v2_pred = np.asarray(results["variant_2_predictions"])
    v2_true = np.asarray(results["variant_2_labels"])

    assert g_pred.shape == g_true.shape
    assert v1_pred.shape == v1_true.shape == g_true.shape
    assert v2_pred.shape == v2_true.shape == g_true.shape

    # ---------- Genotype：全体样本 ----------
    geno_f1 = _safe_f1(g_true, g_pred)
    geno_report = _cls_report(g_true, g_pred)
    geno_cm = _cm(g_true, g_pred)

    # ---------- 触发掩码（GT 与 Pred 两套） ----------
    # GT-gated：Variant-1 在 geno!=0/0；Variant-2 在 geno==1/2
    mask_v1_gt = (g_true != geno_index_00)
    mask_v2_gt = (g_true == geno_index_het_alt)

    # Pred-gated：模拟推理时的实际触发
    mask_v1_pred = (g_pred != geno_index_00)
    mask_v2_pred = (g_pred == geno_index_het_alt)

    # ---------- Variant-1 ----------
    v1_f1_gt   = _safe_f1(v1_true[mask_v1_gt],   v1_pred[mask_v1_gt],   average="macro")
    v1_f1_pred = _safe_f1(v1_true[mask_v1_pred], v1_pred[mask_v1_pred], average="macro")

    v1_report_gt   = _cls_report(v1_true[mask_v1_gt],   v1_pred[mask_v1_gt])
    v1_report_pred = _cls_report(v1_true[mask_v1_pred], v1_pred[mask_v1_pred])

    # 固定 6 类标签顺序，避免子集缺类时矩阵形状变化
    def _cm_fixed(y_true, y_pred):
        if y_true.size == 0:
            return []
        return confusion_matrix(y_true, y_pred, labels=V6_LABELS).tolist()

    v1_cm_gt   = _cm_fixed(v1_true[mask_v1_gt],   v1_pred[mask_v1_gt])
    v1_cm_pred = _cm_fixed(v1_true[mask_v1_pred], v1_pred[mask_v1_pred])

    # 归因：本该触发但没触发（由 genotype 造成的漏评样本）
    v1_should = int(mask_v1_gt.sum())
    v1_did = int(mask_v1_pred.sum())
    v1_missed_by_geno = int(np.logical_and(mask_v1_gt, ~mask_v1_pred).sum())
    v1_spurious_by_geno = int(np.logical_and(~mask_v1_gt, mask_v1_pred).sum())

    # ---------- Variant-2 ----------
    v2_f1_gt   = _safe_f1(v2_true[mask_v2_gt],   v2_pred[mask_v2_gt],   average="macro")
    v2_f1_pred = _safe_f1(v2_true[mask_v2_pred], v2_pred[mask_v2_pred], average="macro")

    v2_report_gt   = _cls_report(v2_true[mask_v2_gt],   v2_pred[mask_v2_gt])
    v2_report_pred = _cls_report(v2_true[mask_v2_pred], v2_pred[mask_v2_pred])

    v2_cm_gt   = _cm_fixed(v2_true[mask_v2_gt],   v2_pred[mask_v2_gt])
    v2_cm_pred = _cm_fixed(v2_true[mask_v2_pred], v2_pred[mask_v2_pred])

    v2_should = int(mask_v2_gt.sum())
    v2_did = int(mask_v2_pred.sum())
    v2_missed_by_geno = int(np.logical_and(mask_v2_gt, ~mask_v2_pred).sum())
    v2_spurious_by_geno = int(np.logical_and(~mask_v2_gt, mask_v2_pred).sum())

    out = {
        "genotype": {
            "f1": geno_f1,
            "report": geno_report,
            "confusion_matrix": geno_cm,
        },
        "variant_1": {
            "f1_gt_gated": v1_f1_gt,
            "f1_pred_gated": v1_f1_pred,
            "report_gt_gated": v1_report_gt,
            "report_pred_gated": v1_report_pred,
            "cm_gt_gated": v1_cm_gt,
            "cm_pred_gated": v1_cm_pred,
            "counts": {
                "should_trigger_by_gt": v1_should,
                "did_trigger_by_pred": v1_did,
                "missed_due_to_genotype": v1_missed_by_geno,
                "spurious_due_to_genotype": v1_spurious_by_geno,
            },
        },
        "variant_2": {
            "f1_gt_gated": v2_f1_gt,
            "f1_pred_gated": v2_f1_pred,
            "report_gt_gated": v2_report_gt,
            "report_pred_gated": v2_report_pred,
            "cm_gt_gated": v2_cm_gt,
            "cm_pred_gated": v2_cm_pred,
            "counts": {
                "should_trigger_by_gt": v2_should,
                "did_trigger_by_pred": v2_did,
                "missed_due_to_genotype": v2_missed_by_geno,
                "spurious_due_to_genotype": v2_spurious_by_geno,
            },
        },
        "meta": {
            "num_samples": int(g_true.size),
            "geno_index_map": {
                "00": geno_index_00, "10": geno_index_het,
                "11": geno_index_hom, "12": geno_index_het_alt
            },
        },
    }

    if verbose:
        def _fmt(x): return "nan" if (x != x) else f"{x:.4f}"
        print("== Head-wise F1 (GT-gated vs Pred-gated) ==")
        print(f"Genotype F1 (all):      { _fmt(geno_f1) }")
        print(f"Variant-1 F1 (GT):      { _fmt(v1_f1_gt) }   | Pred: { _fmt(v1_f1_pred) }   "
              f"| missed_by_geno={v1_missed_by_geno}/{v1_should}")
        print(f"Variant-2 F1 (GT):      { _fmt(v2_f1_gt) }   | Pred: { _fmt(v2_f1_pred) }   "
              f"| missed_by_geno={v2_missed_by_geno}/{v2_should}")
    return out
