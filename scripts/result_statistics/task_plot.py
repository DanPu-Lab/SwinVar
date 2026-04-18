import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import numpy as np
import tables


def plot_pr_task(y_true, y_score, class_names, ax, fig_name, legend_loc="upper right", colors=None):
    n_cls = y_score.shape[1]
    y_true_bin = label_binarize(y_true, classes=np.arange(n_cls))

    precision, recall, ap = {}, {}, {}
    valid_classes = []

    for k in range(n_cls):
        if np.sum(y_true_bin[:, k]) == 0:
            continue
        precision[k], recall[k], _ = precision_recall_curve(y_true_bin[:, k], y_score[:, k])
        ap[k] = average_precision_score(y_true_bin[:, k], y_score[:, k])
        valid_classes.append(k)

    if not valid_classes:
        ax.set_title(f"{fig_name} (No valid class)")
        return

    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin[:, valid_classes].ravel(),
        y_score[:, valid_classes].ravel()
    )
    ap["micro"] = average_precision_score(
        y_true_bin[:, valid_classes], y_score[:, valid_classes], average="micro"
    )
    ap["macro"] = average_precision_score(
        y_true_bin[:, valid_classes], y_score[:, valid_classes], average="macro"
    )

    all_recall = np.unique(np.concatenate([recall[k] for k in valid_classes]))
    prec_interp = np.zeros_like(all_recall)
    for k in valid_classes:
        prec_interp += np.interp(all_recall, recall[k][::-1], precision[k][::-1])
    precision["macro"] = prec_interp / len(valid_classes)
    recall["macro"] = all_recall

    def get_color(idx):
        if colors and idx < len(colors):
            return colors[idx]
        return None

    ax.plot(recall["micro"], precision["micro"], lw=2.5, label=f"Micro (AP={ap['micro']:.3f})", color=get_color(0))
    ax.plot(recall["macro"], precision["macro"], lw=2.0, label=f"Macro (mAP={ap['macro']:.3f})", color=get_color(1))

    for i, k in enumerate(valid_classes):
        ax.plot(
            recall[k],
            precision[k],
            lw=1,
            alpha=0.75,
            label=f"{class_names[k]} (AP={ap[k]:.2f})",
            color=get_color(i + 2)
        )

    ax.set_xlabel("Recall", fontsize=8)
    ax.set_ylabel("Precision", fontsize=8)
    ax.set_title(fig_name, fontsize=10)
    ax.tick_params(axis='both', labelsize=6)
    ax.grid(True, linestyle='--', linewidth=0.5)

    ax.legend(
        fontsize=7,
        loc=legend_loc,
        frameon=True,
        fancybox=True,
        framealpha=0.5,
        borderpad=0.4,
        labelspacing=0.3,
        handlelength=1.5
    )


def run(file_path, save_file, color_list=None):
    tables_data = tables.open_file(file_path, "r")
    y_score_genotype = tables_data.root.Score_Genotype[:]
    y_true_genotype = tables_data.root.Labels_Genotype[:]
    y_score_variant_1 = tables_data.root.Score_Variant_1[:]
    y_true_variant_1 = tables_data.root.Labels_Variant_1[:]
    y_score_variant_2 = tables_data.root.Score_Variant_2[:]
    y_true_variant_2 = tables_data.root.Labels_Variant_2[:]
    tables_data.close()

    fig, axs = plt.subplots(3, 1, figsize=(3, 8))

    plot_pr_task(
        y_true_genotype,
        y_score_genotype,
        class_names=["0", "1", "2", "3"],
        ax=axs[0],
        fig_name="Task1 Genotype",
        legend_loc="lower left",
        colors=color_list
    )
    plot_pr_task(
        y_true_variant_1,
        y_score_variant_1,
        class_names=["A", "C", "G", "T", "Insert", "Deletion"],
        ax=axs[1],
        fig_name="Task2 Variant_1",
        legend_loc="lower left",
        colors=color_list
    )
    plot_pr_task(
        y_true_variant_2,
        y_score_variant_2,
        class_names=["A", "C", "G", "T", "Insert", "Deletion"],
        ax=axs[2],
        fig_name="Task3 Variant_2",
        legend_loc="upper right",
        colors=color_list
    )

    plt.tight_layout()
    plt.savefig(save_file, dpi=600)
    plt.close()


if __name__ == "__main__":
    # 🎨 可自定义颜色序列
    color_list = [
        "#4B0082",  # Micro
        "#1E90FF",  # Macro
        "#008080", "#FF8C00", "#8A2BE2", "#2E8B57", "#DC143C", "#4682B4"  # Classes
    ]

    run(
        file_path="/data2/lijie/result/Transformer_pileup_3_channel/HG003_4_5to2_WES/train_moe/balance_2/pr_data.h5",
        save_file="/data2/lijie/result/Transformer_pileup_3_channel/pr/HG002.svg",
        color_list=color_list
    )

    run(
        file_path="/data2/lijie/result/Transformer_pileup_3_channel/HG002_4_5to3_WES/train_moe/balance_2/pr_data.h5",
        save_file="/data2/lijie/result/Transformer_pileup_3_channel/pr/HG003.svg",
        color_list=color_list
    )

    run(
        file_path="/data2/lijie/result/Transformer_pileup_3_channel/HG002_3_5to4_WES/train_moe/balance_2/pr_data.h5",
        save_file="/data2/lijie/result/Transformer_pileup_3_channel/pr/HG004.svg",
        color_list=color_list
    )

    run(
        file_path="/data2/lijie/result/Transformer_pileup_3_channel/HG002_3_4to5_WES/train_moe/balance_2/pr_data.h5",
        save_file="/data2/lijie/result/Transformer_pileup_3_channel/pr/HG005.svg",
        color_list=color_list
    )
