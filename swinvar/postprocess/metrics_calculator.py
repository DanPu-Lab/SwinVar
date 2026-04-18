import pandas as pd
import numpy as np

from swinvar.preprocess.parameters import VARIANT_LABELS
from swinvar.postprocess.vcf_generator import get_vcf


def variant_df(chrom, pos, ref, indel_info, variant_1_predictions, variant_1_labels, variant_2_predictions, variant_2_labels, genotype_predictions, genotype_labels, bam=None, output_vcf=None, reference=None):

    reverse_VARIANT_LABELS = {v: k for k, v in VARIANT_LABELS.items()}

    df = pd.DataFrame({"CHROM": chrom, "POS": pos, "REF": ref, "INDEL": indel_info, "Prediction_Genotype": genotype_predictions, "Label_Genotype": genotype_labels, "Prediction_1": variant_1_predictions, "Label_1": variant_1_labels, "Prediction_2": variant_2_predictions, "Label_2": variant_2_labels})

    
    df["REF_BASE"] = df["REF"].str[0]

    df[["INDEL_1", "INDEL_2"]] = df["INDEL"].str.split("|", expand=True).iloc[:, :2].fillna("")
    indel_condition = [
        (df["Prediction_1"].isin(["Indel", "Deletion"])) & (df["Prediction_2"].isin(["Indel", "Deletion"])),
        (df["Prediction_1"].isin(["Indel", "Deletion"])) | (df["Prediction_2"].isin(["Indel", "Deletion"])),
    ]
    indel_choices = [
        df["INDEL_1"] + "," + df["INDEL_2"],
        df["INDEL_1"],
    ]
    df["ALT"] = np.select(indel_condition, indel_choices, default="")


    df["Prediction_1"] = df["Prediction_1"].map(reverse_VARIANT_LABELS)
    df["Prediction_2"] = df["Prediction_2"].map(reverse_VARIANT_LABELS)
    df["Label_1"] = df["Label_1"].map(reverse_VARIANT_LABELS)
    df["Label_2"] = df["Label_2"].map(reverse_VARIANT_LABELS)

    df["Prediction"] = df["Prediction_1"] + df["Prediction_2"]
    df["Label"] = df["Label_1"] + df["Label_2"]

    metrics_conditions = [
        (df["Label_Genotype"] == 0) & (df["Prediction_Genotype"] == 0),  # TN
        (df["Label_Genotype"] != df["Prediction_Genotype"]) & (df["Prediction_Genotype"] != 0),  # FP
        (df["Label_Genotype"] != df["Prediction_Genotype"]) & (df["Prediction_Genotype"] == 0),  # FN

        (df["Label_Genotype"] == 1) & (df["Prediction_1"] == df["Label_1"]),  # TP
        (df["Label_Genotype"] == 1) & (df["Prediction_1"] != df["Label_1"]) & (df["Prediction_1"] != df["REF_BASE"]),  # FP
        (df["Label_Genotype"] == 1) & (df["Prediction_1"] != df["Label_1"]) & (df["Prediction_1"] == df["REF_BASE"]),  # FN

        (df["Label_Genotype"] == 2) & (df["Prediction_1"] == df["Label_1"]),  # TP
        (df["Label_Genotype"] == 2) & (df["Prediction_1"] != df["Label_1"]) & (df["Prediction"] != df["REF_BASE"]),  # FP
        (df["Label_Genotype"] == 2) & (df["Prediction_1"] != df["Label_1"]) & (df["Prediction"] == df["REF_BASE"]),  # FN

        (df["Label_Genotype"] == 3) & (df["Prediction"] == df["Label"]),  # TP
        (df["Label_Genotype"] == 3) & (df["Prediction"] != df["Label"]) & (df["Prediction"] != df["REF"]),  # FP
        (df["Label_Genotype"] == 3) & (df["Prediction"] != df["Label"]) & (df["Prediction"] == df["REF"]),  # FN
    ]
    metrics_choices = ["True negative", "False positive", "False negative", "True positive", "False positive", "False negative", "True positive", "False positive", "False negative", "True positive", "False positive", "False negative"]
    df["Variant"] = np.select(metrics_conditions, metrics_choices, default="Unknown")

    if output_vcf is not None:
        get_vcf(df.copy(), bam, output_vcf, reference)

    return df


def calculate_metrics(df):

    variants_type = ["True negative", "False positive", "True positive", "False negative"]
    
    variant_dict = df["Variant"].value_counts().reindex(variants_type, fill_value=0).to_dict()
    variant_metrics = {k: int(v) for k, v in variant_dict.items()}

    sensitivity = variant_metrics["True positive"] / (variant_metrics["True positive"] + variant_metrics["False negative"]) if (variant_metrics["True positive"] + variant_metrics["False negative"]) != 0 else 0
    precision = variant_metrics["True positive"] / (variant_metrics["True positive"] + variant_metrics["False positive"]) if (variant_metrics["True positive"] + variant_metrics["False positive"]) != 0 else 0
    f1_score = 2 * (sensitivity * precision) / (sensitivity + precision) if (sensitivity + precision) != 0 else 0

    return variant_metrics, sensitivity, precision, f1_score
