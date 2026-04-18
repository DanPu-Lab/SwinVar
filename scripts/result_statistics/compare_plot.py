from cyvcf2 import VCF
import pandas as pd
import matplotlib.pyplot as plt
import os

def normalize_gt(gt_tuple):
    if -1 in gt_tuple:
        return "./."
    return "/".join(map(str, sorted(gt_tuple)))

def load_vcf_variants(vcf_path):
    snp_dict = dict()
    indel_dict = dict()
    gt_dict = dict()

    for var in VCF(vcf_path):
        chrom = var.CHROM.replace("chr", "")
        pos = var.POS
        ref = var.REF
        alt_list = var.ALT

        if any("<" in alt or ">" in alt for alt in alt_list):
            continue  # 忽略结构突变或缺失ALT的记录

        key = (chrom, pos, ref.upper())
        alts = tuple(sorted([alt.upper() for alt in alt_list]))
        gt = normalize_gt(var.genotypes[0][:2])

        if len(ref) == 1 and all(len(alt) == 1 for alt in alt_list):
            snp_dict[key] = alts
        else:
            indel_dict[key] = alts

        gt_dict[key] = gt

    return snp_dict, indel_dict, gt_dict

def evaluate(pred, truth, gt_pred, gt_truth):
    tp = fp = fn = 0

    for key in pred:
        pred_alt = set(pred[key])
        pred_gt = gt_pred.get(key, "./.")
        truth_alt = set(truth.get(key, []))
        truth_gt = gt_truth.get(key, "./.")

        if key in truth and pred_alt == truth_alt and pred_gt == truth_gt:
            tp += 1
        else:
            fp += 1

    for key in truth:
        truth_alt = set(truth[key])
        truth_gt = gt_truth.get(key, "./.")
        pred_alt = set(pred.get(key, []))
        pred_gt = gt_pred.get(key, "./.")

        if key not in pred or pred_alt != truth_alt or pred_gt != truth_gt:
            fn += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

    return recall, precision, f1

def collect_metrics(vcf_truth, tool_vcf_paths):
    snp_scores = {}
    indel_scores = {}

    truth_snp, truth_indel, gt_truth = load_vcf_variants(vcf_truth)

    for tool_name, pred_path in tool_vcf_paths.items():
        pred_snp, pred_indel, gt_pred = load_vcf_variants(pred_path)
        snp_scores[tool_name] = evaluate(pred_snp, truth_snp, gt_pred, gt_truth)
        indel_scores[tool_name] = evaluate(pred_indel, truth_indel, gt_pred, gt_truth)

    return snp_scores, indel_scores

def save_metrics_to_csv(snp_scores, indel_scores, out_path):
    rows = []
    for tool, (r, p, f1) in snp_scores.items():
        rows.append({"Tool": tool, "Type": "SNP", "Recall": r, "Precision": p, "F1": f1})
    for tool, (r, p, f1) in indel_scores.items():
        rows.append({"Tool": tool, "Type": "INDEL", "Recall": r, "Precision": p, "F1": f1})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

def load_metrics_from_csv(file_path):
    df = pd.read_csv(file_path)
    snp_df = df[df["Type"] == "SNP"]
    indel_df = df[df["Type"] == "INDEL"]
    return snp_df, indel_df

def plot_combined_metrics_from_file(csv_path, save_path, colors=None):
    snp_df, indel_df = load_metrics_from_csv(csv_path)

    tools = snp_df["Tool"].tolist()
    x = list(range(len(tools)))

    default_colors = {
        "Recall": "blue",
        "Precision": "green",
        "F1": "red"
    }
    if colors is not None:
        default_colors.update(colors)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

    for ax, df, title in zip(axes, [snp_df, indel_df], ["SNP", "INDEL"]):
        ax.scatter(x, df["Recall"], label="Recall", marker="o", s=10, color=default_colors["Recall"])
        ax.scatter(x, df["Precision"], label="Precision", marker="^", s=10, color=default_colors["Precision"])
        ax.scatter(x, df["F1"], label="F1-score", marker="s", s=10, color=default_colors["F1"])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Score', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(df["Tool"], rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='both', labelsize=6)
        ax.grid(True, linestyle='--', alpha=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=7, handletextpad=0.1)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=600)
    plt.show()

if __name__ == "__main__":
    vcf_truth = "/data2/lijie/data/hs37d5/HG003/HG003.vcf"
    tool_vcfs = {
        "SwinVar_2_pre": "/data2/lijie/result/Transformer_pileup_3_channel/HG002_4_5to3_WES/output_2_pre.vcf",
        "SwinVar_2": "/data2/lijie/result/Transformer_pileup_3_channel/HG002_4_5to3_WES/output_2.vcf",
        "SwinVar_5_pre": "/data2/lijie/result/Transformer_pileup_3_channel/HG002_4_5to3_WES/output_5_pre.vcf",
        "SwinVar_5": "/data2/lijie/result/Transformer_pileup_3_channel/HG002_4_5to3_WES/output_5.vcf",
        "SwinVar_10_pre": "/data2/lijie/result/Transformer_pileup_3_channel/HG002_4_5to3_WES/output_10_pre.vcf",
        "SwinVar_10": "/data2/lijie/result/Transformer_pileup_3_channel/HG002_4_5to3_WES/output_10.vcf",
        "SwinVar_20_pre": "/data2/lijie/result/Transformer_pileup_3_channel/HG002_4_5to3_WES/output_20_pre.vcf",
        "SwinVar_20": "/data2/lijie/result/Transformer_pileup_3_channel/HG002_4_5to3_WES/output_20.vcf",
        "DeepVariant": "/data2/lijie/result/deepvariant/HG003.vcf",
        "Clair3": "/data2/lijie/result/clair3/merge_output.vcf",
        "GATK": "/data2/lijie/result/gatk/HG003.vcf",
        "Strelka": "/data2/lijie/script/Tools/strelka-2.9.10.centos6_x86_64/result/results/variants/variants.vcf.gz",

    }

    metrics_csv = "/data2/lijie/result/snp_indel_metrics.csv"
    output_fig = "/data2/lijie/result/snp_indel_combined.svg"

    snp_scores, indel_scores = collect_metrics(vcf_truth, tool_vcfs)
    save_metrics_to_csv(snp_scores, indel_scores, metrics_csv)

    colors = {
        "Recall": "#B777D5",
        "Precision": "#61D6D6",
        "F1": "#65D063"
    }

    metrics_csv = "/data2/lijie/result/snp_indel_metrics_update.csv"
    plot_combined_metrics_from_file(metrics_csv, output_fig, colors)
