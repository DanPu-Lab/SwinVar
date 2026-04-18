import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

import pandas as pd
import numpy as np

from swinvar.postprocess.metrics_calculator import calculate_metrics
from swinvar.preprocess.parameters import VARIANT, GENOTYPE_LABELS
from swinvar.preprocess.utils import check_directory


def task_performance(file_path, save_path):

    df = pd.read_csv(file_path)
    # test_df_snp = df[~df["Label"].str.contains("Insert|Deletion", regex=True)]
    # test_df_indel = df[df["Label"].str.contains("Insert|Deletion", regex=True)]

    # Task1: Genotype
    task_1_conditions = [
        (df["Label_Genotype"] == 0) & (df["Prediction_Genotype"] == 0),  # TN
        (df["Label_Genotype"] != 0) & (df["Prediction_Genotype"] == 0),  # FN
        (df["Label_Genotype"] == df["Prediction_Genotype"]),  # TP
        (df["Label_Genotype"] != df["Prediction_Genotype"]),  # FP
    ]
    task_1_choices = ["True negative", "False negative", "True positive", "False positive"]
    df["Variant"] = np.select(task_1_conditions, task_1_choices, default="Unknown")

    # print(df[df["Variant"] == "Unknown"])

    task_1_metrics, task_1_sensitivity, task_1_precision, task_1_f1_score = calculate_metrics(df)

    # Task2: Variant 1
    task_2_df = df[(df["Label_Genotype"] != 0)]
    task_2_conditions = [
        (task_2_df["Label_1"] == task_2_df["Prediction_1"]),  # TP
        (task_2_df["Label_1"] != task_2_df["Prediction_1"]) & (task_2_df["Prediction_1"] == task_2_df["REF_BASE"]),  # FN
        (task_2_df["Label_1"] != task_2_df["Prediction_1"]) & (task_2_df["Prediction_1"] != task_2_df["REF_BASE"]),  # FP
    ]
    task_2_choices = ["True positive", "False negative", "False positive"]
    task_2_df.loc[:, "Variant"] = np.select(task_2_conditions, task_2_choices, default="Unknown")
    task_2_metrics, task_2_sensitivity, task_2_precision, task_2_f1_score = calculate_metrics(task_2_df)


    # Task3: Variant 2
    task_3_df = df[(df["Label_Genotype"] == 3)]
    task_3_conditions = [
        (task_3_df["Label_2"] == task_3_df["Prediction_2"]),  # TP
        (task_3_df["Label_2"] != task_3_df["Prediction_2"]) & (task_3_df["Prediction_2"] == task_3_df["REF_BASE"]),  # FN
        (task_3_df["Label_2"] != task_3_df["Prediction_2"]) & (task_3_df["Prediction_2"] != task_3_df["REF_BASE"]),  # FP
    ]
    task_3_choices = ["True positive", "False negative", "False positive"]
    task_3_df.loc[:, "Variant"] = np.select(task_3_conditions, task_3_choices, default="Unknown")
    task_3_metrics, task_3_sensitivity, task_3_precision, task_3_f1_score = calculate_metrics(task_3_df)


    # result = pd.DataFrame({
    #     "X": ["Recall", "Precision", "F1 score"],
    #     "Task 1": [task_1_sensitivity, task_1_precision, task_1_f1_score],
    #     "Task 2": [task_2_sensitivity, task_2_precision, task_2_f1_score],
    #     "Task 3": [task_3_sensitivity, task_3_precision, task_3_f1_score],
    # })
    result = pd.DataFrame({
        "X": ["Task 1", "Task 2", "Task 3"],
        "Recall": [task_1_sensitivity, task_2_sensitivity, task_3_sensitivity],
        "Precision": [task_1_precision, task_2_precision, task_3_precision],
        "F1 score": [task_1_f1_score, task_2_f1_score, task_3_f1_score],
    })
    result.to_csv(save_path, index=False)

    # metrics_conditions = [
    #     (df["Label_Genotype"] == 0) & (df["Prediction_Genotype"] == 0),  # TN
    #     (df["Label_Genotype"] != df["Prediction_Genotype"]) & (df["Prediction_Genotype"] != 0),  # FP
    #     (df["Label_Genotype"] != df["Prediction_Genotype"]) & (df["Prediction_Genotype"] == 0),  # FN

    #     (df["Label_Genotype"] == 1) & (df["Prediction_1"] == df["Label_1"]),  # TP
    #     (df["Label_Genotype"] == 1) & (df["Prediction_1"] != df["Label_1"]) & (df["Prediction_1"] != df["REF_BASE"]),  # FP
    #     (df["Label_Genotype"] == 1) & (df["Prediction_1"] != df["Label_1"]) & (df["Prediction_1"] == df["REF_BASE"]),  # FN

    #     (df["Label_Genotype"] == 2) & (df["Prediction_1"] == df["Label_1"]),  # TP
    #     (df["Label_Genotype"] == 2) & (df["Prediction_1"] != df["Label_1"]) & (df["Prediction"] != df["REF_BASE"]),  # FP
    #     (df["Label_Genotype"] == 2) & (df["Prediction_1"] != df["Label_1"]) & (df["Prediction"] == df["REF_BASE"]),  # FN

    #     (df["Label_Genotype"] == 3) & (df["Prediction"] == df["Label"]),  # TP
    #     (df["Label_Genotype"] == 3) & (df["Prediction"] != df["Label"]) & (df["Prediction"] != df["REF"]),  # FP
    #     (df["Label_Genotype"] == 3) & (df["Prediction"] != df["Label"]) & (df["Prediction"] == df["REF"]),  # FN
    # ]
    # metrics_choices = ["True negative", "False positive", "False negative", "True positive", "False positive", "False negative", "True positive", "False positive", "False negative", "True positive", "False positive", "False negative"]

    # df["Variant"] = np.select(metrics_conditions, metrics_choices, default="Unknown")
    # print(df[df["Variant"] == "Unknown"])

    # variant_metrics, sensitivity, precision, f1_score = calculate_metrics(df)
    # print(variant_metrics, sensitivity, precision, f1_score)


def heatmap_matrix(file_path, save_path):

    df = pd.read_csv(file_path)

    task_1 = df.copy()
    task_1["Label_Genotype"] = task_1["Label_Genotype"].astype(
        pd.CategoricalDtype(categories=GENOTYPE_LABELS.values(), ordered=True)
    )
    task_1["Prediction_Genotype"] = task_1["Prediction_Genotype"].astype(
        pd.CategoricalDtype(categories=GENOTYPE_LABELS.values(), ordered=True)
    )
    task_1_matrix = pd.crosstab(
        task_1["Label_Genotype"],
        task_1["Prediction_Genotype"],
        dropna=False
    )

    task_2 = df[df["Label_Genotype"] != 0].copy()
    task_2["Label_1"] = task_2["Label_1"].astype(
        pd.CategoricalDtype(categories=VARIANT, ordered=True)
    )
    task_2["Prediction_1"] = task_2["Prediction_1"].astype(
        pd.CategoricalDtype(categories=VARIANT, ordered=True)
    )
    task_2_matrix = pd.crosstab(
        task_2["Label_1"],
        task_2["Prediction_1"],
        dropna=False
    )

    task_3= df[df["Label_Genotype"] == 3].copy()
    task_3["Label_2"] = task_3["Label_2"].astype(
        pd.CategoricalDtype(categories=VARIANT, ordered=True)
    )
    task_3["Prediction_2"] = task_3["Prediction_2"].astype(
        pd.CategoricalDtype(categories=VARIANT, ordered=True)
    )
    task_3_matrix = pd.crosstab(
        task_3["Label_2"],
        task_3["Prediction_2"],
        dropna=False
    )

    task_1_matrix_log = np.log10(task_1_matrix + 1)
    task_2_matrix_log = np.log10(task_2_matrix + 1)
    task_3_matrix_log = np.log10(task_3_matrix + 1)

    task_1_matrix_log.to_csv(f"{save_path}_heatmap_task_1.csv")
    task_2_matrix_log.to_csv(f"{save_path}_heatmap_task_2.csv")
    task_3_matrix_log.to_csv(f"{save_path}_heatmap_task_3.csv")
    # print(matrix)
    # df["Label_Genotype"] == 0) & (df["Prediction_Genotype"] == 0
    # pass

if __name__ == "__main__":

    root_path = "/data2/lijie/result/Transformer_pileup_3_channel"

    HG002 = "/data2/lijie/result/Transformer_pileup_3_channel/HG003_4_5to2_WES/train_moe/balance_2/test_df.csv"
    HG003 = "/data2/lijie/result/Transformer_pileup_3_channel/HG002_4_5to3_WES/train_moe/balance_2/test_df.csv"
    HG004 = "/data2/lijie/result/Transformer_pileup_3_channel/HG002_3_5to4_WES/train_moe/balance_2/test_df.csv"
    HG005 = "/data2/lijie/result/Transformer_pileup_3_channel/HG002_3_4to5_WES/train_moe/balance_2/test_df.csv"

    task = True
    heatmap = True

    if task:
        save_path = os.path.join(root_path, "task_performance")
        check_directory(save_path)

        task_performance(HG002,
            os.path.join(save_path, "HG002_task.csv"))

        task_performance(HG003,
            os.path.join(save_path, "HG003_task.csv"))

        task_performance(HG004,
            os.path.join(save_path, "HG004_task.csv"))

        task_performance(HG005,
            os.path.join(save_path, "HG005_task.csv"))

    if heatmap:
        save_path = os.path.join(root_path, "heatmap")
        check_directory(save_path)
        heatmap_matrix(HG002, os.path.join(save_path, "HG002"))
        heatmap_matrix(HG003, os.path.join(save_path, "HG003"))
        heatmap_matrix(HG004, os.path.join(save_path, "HG004"))
        heatmap_matrix(HG005, os.path.join(save_path, "HG005"))
