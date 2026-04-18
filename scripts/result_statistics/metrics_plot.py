import pandas as pd
import os
import re

from swinvar.preprocess.utils import check_directory
from scripts.result_statistics.plot import combined_plot


def read_result(file_path, output_path):

    check_directory(output_path)

    snp_recall_data = []
    indel_recall_data = []

    snp_precision_data = []
    indel_precision_data = []

    snp_f1_data = []
    indel_f1_data = []
    epoch = 1

    with open(file_path, "r") as f:
        line = f.readline()
        while line:
            if line.startswith("SNP") or line.startswith("INDEL"):
                variant_type = line.strip()
                f.readline()
                f.readline()
                train_metrics_line = f.readline()
                train_match = re.findall(r"(\w+):([\d.]+)", train_metrics_line)
                train_metric_dict = {k: float(v) for k, v in train_match}

                val_metrics_line = f.readline()
                val_match = re.findall(r"(\w+):([\d.]+)", val_metrics_line)
                val_metric_dict = {k: float(v) for k, v in val_match}

                if variant_type.startswith("SNP"):
                    snp_recall_data.append({"Epoch": epoch, "Train": train_metric_dict["Sensitivity"], "Val": val_metric_dict["Sensitivity"]})
                    snp_precision_data.append({"Epoch": epoch, "Train": train_metric_dict["Precision"], "Val": val_metric_dict["Precision"]})
                    snp_f1_data.append({"Epoch": epoch, "Train": train_metric_dict["Score"], "Val": val_metric_dict["Score"]})
                elif variant_type.startswith("INDEL"):
                    indel_recall_data.append({"Epoch": epoch, "Train": train_metric_dict["Sensitivity"], "Val": val_metric_dict["Sensitivity"]})
                    indel_precision_data.append({"Epoch": epoch, "Train": train_metric_dict["Precision"], "Val": val_metric_dict["Precision"]})
                    indel_f1_data.append({"Epoch": epoch, "Train": train_metric_dict["Score"], "Val": val_metric_dict["Score"]})
                    epoch += 1

                if epoch > 90:
                    break
            line = f.readline()


    df_snp_recall = pd.DataFrame(snp_recall_data)
    df_snp_precision = pd.DataFrame(snp_precision_data)
    df_snp_f1 = pd.DataFrame(snp_f1_data)
    df_indel_recall = pd.DataFrame(indel_recall_data)
    df_indel_precision = pd.DataFrame(indel_precision_data)
    df_indel_f1 = pd.DataFrame(indel_f1_data)


    df_snp_recall.to_csv(os.path.join(output_path, "SNP_recall.csv"), index=False)
    df_snp_precision.to_csv(os.path.join(output_path, "SNP_precision.csv"), index=False)
    df_snp_f1.to_csv(os.path.join(output_path, "SNP_f1.csv"), index=False)
    df_indel_recall.to_csv(os.path.join(output_path, "INDEL_recall.csv"), index=False)
    df_indel_precision.to_csv(os.path.join(output_path, "INDEL_precision.csv"), index=False)
    df_indel_f1.to_csv(os.path.join(output_path, "INDEL_f1.csv"), index=False)

    combined_plot(df_snp_recall, df_snp_precision, df_snp_f1, df_indel_recall, df_indel_precision, df_indel_f1, os.path.join(output_path, "metrics.svg"))
    # combined_plot(df_snp_recall, df_snp_precision, df_snp_f1, df_indel_recall, df_indel_precision, df_indel_f1, os.path.join(output_path, "metrics.jpg"))
    

if __name__ == "__main__":
    file_path = "/data2/lijie/result/Transformer_pileup_3_channel/HG003_4_5to2_WES/train_moe/balance_2/train_log.txt"
    output_path = "/data2/lijie/result/Transformer_pileup_3_channel/HG003_4_5to2_WES/train_moe/balance_2/metrics"
    read_result(file_path, output_path)