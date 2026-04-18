import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import matplotlib.gridspec as gridspec

# 配色
colors = {
    'Recall': '#3D3B8B',
    'Precision': '#927BBC',
    'F1 score': '#E5CBE1'
}

# 折线图函数
def plot_metric(ax, df, title, ylabel):
    ax.plot(df['Epoch'], df['Train'], label='Train', linewidth=1.5, color='#927BBC')
    val_line, = ax.plot(df['Epoch'], df['Val'], label='Validation', linewidth=1.5, color='#3D3B8B')
    max_idx = df['Val'].idxmax()
    ax.scatter(df.loc[max_idx, 'Epoch'], df.loc[max_idx, 'Val'],
               color=val_line.get_color(), s=8, zorder=3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
    ax.grid(True, linestyle='--', linewidth=0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

# 柱状图函数
def plot_task_bar(ax, file):
    df = pd.read_csv(file)
    tasks = df['X'].values
    num_tasks = len(tasks)
    x = np.linspace(0, 0.6 * (num_tasks - 1), num_tasks)
    bar_width = 0.12
    offsets = [-bar_width, 0, bar_width]

    for idx, metric in enumerate(['Recall', 'Precision', 'F1 score']):
        values = df[metric].values
        ax.bar(x + offsets[idx], values,
               width=bar_width,
               label=metric,
               color=colors[metric],
               edgecolor='black',
               linewidth=0.7)

    sample_name = os.path.basename(file).split('_')[0]
    ax.set_title(sample_name, fontsize=10)
    ax.set_ylabel("Score", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=8)
    ax.set_ylim(0, 1.1)

    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(axis='x', direction='out', length=5, width=1.2, colors='black', labelsize=6)
    ax.tick_params(axis='y', direction='out', length=5, width=1.2, colors='black', labelsize=6)

# 主函数
def combined_plot(
    recall_snv, precision_snv, f1_snv,
    recall_indel, precision_indel, f1_indel,
    save_path
):
    fig = plt.figure(figsize=(5, 11))
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.5, wspace=0.5)

    # 折线图（前3行）
    axs = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    plot_metric(axs[0], recall_snv, "SNP", "Recall")
    plot_metric(axs[1], recall_indel, "INDEL", "Recall")
    plot_metric(axs[2], precision_snv, "SNP", "Precision")
    plot_metric(axs[3], precision_indel, "INDEL", "Precision")
    plot_metric(axs[4], f1_snv, "SNP", "F1 Score")
    plot_metric(axs[5], f1_indel, "INDEL", "F1 Score")

    # 柱状图（后2行）
    folder_path = '/data2/lijie/result/Transformer_pileup_3_channel/task_performance'
    csv_files = sorted(glob.glob(os.path.join(folder_path, 'HG00*_task.csv')))
    ax7 = fig.add_subplot(gs[3, 0])
    ax8 = fig.add_subplot(gs[3, 1])
    ax9 = fig.add_subplot(gs[4, 0])
    ax10 = fig.add_subplot(gs[4, 1])

    for ax, file in zip([ax7, ax8, ax9, ax10], csv_files):
        plot_task_bar(ax, file)

    # 图注标注
    fig.text(0.03, 0.975, 'a', fontsize=12, fontweight='bold')
    fig.text(0.03, 0.395, 'b', fontsize=12, fontweight='bold')

    # 图例
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[key], edgecolor='black') for key in colors]
    labels = list(colors.keys())
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True, fontsize=7)

    # fig.subplots_adjust(left=0.06, right=0.98, top=0.97, bottom=0.08, hspace=0.8, wspace=0.4)
    fig.subplots_adjust(top=0.97, bottom=0.048)
    # fig.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(save_path, dpi=600)
    plt.close()
