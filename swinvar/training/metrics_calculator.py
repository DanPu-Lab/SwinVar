import torch
import numpy as np

from swinvar.postprocess.metrics_calculator import variant_df, calculate_metrics


class MetricsCalculator:
    """指标计算类"""
    
    def __init__(self):
        self.train_metrics = {}
        self.val_metrics = {}
        
    def calculate_batch_metrics(self, predictions, labels):
        """计算批次指标"""
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy, correct, total
    
    def process_predictions(self, batch_data, predictions, is_training=True):
        """处理预测结果"""
        variant_1_labels, variant_2_labels, genotype_labels, chrom, pos, ref, indel_info = batch_data
        
        variant_1_pred = torch.argmax(predictions[0], dim=-1)
        variant_2_pred = torch.argmax(predictions[1], dim=-1)
        genotype_pred = torch.argmax(predictions[2], dim=-1)
        
        # 转换为numpy数组
        predictions_data = {
            'variant_1': variant_1_pred.cpu().numpy(),
            'variant_2': variant_2_pred.cpu().numpy(),
            'genotype': genotype_pred.cpu().numpy(),
        }
        
        labels_data = {
            'variant_1': variant_1_labels.cpu().numpy(),
            'variant_2': variant_2_labels.cpu().numpy(),
            'genotype': genotype_labels.cpu().numpy(),
        }
        
        metadata = {
            'chrom': chrom,
            'pos': pos,
            'ref': ref,
            'indel_info': indel_info,
        }
        
        return predictions_data, labels_data, metadata
    
    def calculate_epoch_metrics(self, predictions_list, labels_list, metadata_list, is_training=True):
        """计算epoch指标"""
        # 合并所有批次的数据
        all_variant_1_pred = np.concatenate([p['variant_1'] for p in predictions_list])
        all_variant_2_pred = np.concatenate([p['variant_2'] for p in predictions_list])
        all_genotype_pred = np.concatenate([p['genotype'] for p in predictions_list])
        
        all_variant_1_labels = np.concatenate([l['variant_1'] for l in labels_list])
        all_variant_2_labels = np.concatenate([l['variant_2'] for l in labels_list])
        all_genotype_labels = np.concatenate([l['genotype'] for l in labels_list])
        
        all_chrom = np.concatenate([m['chrom'] for m in metadata_list])
        all_pos = np.concatenate([m['pos'] for m in metadata_list])
        all_ref = np.concatenate([m['ref'] for m in metadata_list])
        all_indel_info = np.concatenate([m['indel_info'] for m in metadata_list])
        
        # 创建DataFrame
        df = variant_df(
            all_chrom, all_pos, all_ref, all_indel_info,
            all_variant_1_pred, all_variant_1_labels,
            all_variant_2_pred, all_variant_2_labels,
            all_genotype_pred, all_genotype_labels
        )
        
        # 分离SNP和INDEL
        df_snp = df[~df["Label"].str.contains("Insert|Deletion", regex=True)]
        df_indel = df[df["Label"].str.contains("Insert|Deletion", regex=True)]
        df = df.drop_duplicates().reset_index(drop=True)
        
        # 计算指标
        variant_count = (df["Label_Genotype"] != 0).sum()
        variant_metrics, sensitivity, precision, f1_score = calculate_metrics(df)
        
        # SNP指标
        variant_snp, sensitivity_snp, precision_snp, f1_score_snp = calculate_metrics(df_snp)
        
        # INDEL指标
        variant_indel, sensitivity_indel, precision_indel, f1_score_indel = calculate_metrics(df_indel)
        
        metrics = {
            'variant_count': variant_count,
            'sensitivity': sensitivity,
            'precision': precision,
            'f1_score': f1_score,
            'variant_metrics': variant_metrics,
            'snp_sensitivity': sensitivity_snp,
            'snp_precision': precision_snp,
            'snp_f1_score': f1_score_snp,
            'snp_variant_metrics': variant_snp,
            'indel_sensitivity': sensitivity_indel,
            'indel_precision': precision_indel,
            'indel_f1_score': f1_score_indel,
            'indel_variant_metrics': variant_indel,
        }
        
        if is_training:
            self.train_metrics = metrics
        else:
            self.val_metrics = metrics
            
        return metrics
    
    def get_metrics_summary(self):
        """获取指标摘要"""
        return {
            'train': self.train_metrics,
            'val': self.val_metrics
        }
    
    def log_metrics(self, logger, epoch, total_epochs, train_metrics, val_metrics, patience_counter):
        """记录指标"""
        
        logger.info(f"Epoch [{epoch+1}/{total_epochs}]")
        logger.info(
            f"Train Loss: {train_metrics["loss"]:.6f}\t"
            f"Train Genotype Acc: {train_metrics['genotype_acc']:.6f}\t"
            f"Train Variant 1 Acc: {train_metrics['variant_1_acc']:.6f}\t"
            f"Train Variant 2 Acc: {train_metrics['variant_2_acc']:.6f}"
        )
        logger.info(
            f"Val   Loss: {val_metrics["loss"]:.6f}\t"
            f"Val Genotype Acc: {val_metrics['genotype_acc']:.6f}\t"
            f"Val Variant 1 Acc: {val_metrics['variant_1_acc']:.6f}\t"
            f"Val Variant 2 Acc: {val_metrics['variant_2_acc']:.6f}"
        )
        logger.info(f"patience: {patience_counter}")
        
        logger.info(f"Train----Total number of variants: {train_metrics['variant_count']}")
        logger.info(f"Val------Total number of variants: {val_metrics['variant_count']}")
        
        # 记录整体指标
        self._log_variant_metrics(logger, train_metrics['variant_metrics'], 
                                 train_metrics['sensitivity'], train_metrics['precision'], 
                                 train_metrics['f1_score'], "Train")
        self._log_variant_metrics(logger, val_metrics['variant_metrics'], 
                                 val_metrics['sensitivity'], val_metrics['precision'], 
                                 val_metrics['f1_score'], "Val")
        
        # 记录SNP指标
        logger.info(f"SNP:")
        self._log_variant_metrics(logger, train_metrics['snp_variant_metrics'], 
                                 train_metrics['snp_sensitivity'], train_metrics['snp_precision'], 
                                 train_metrics['snp_f1_score'], "Train")
        self._log_variant_metrics(logger, val_metrics['snp_variant_metrics'], 
                                 val_metrics['snp_sensitivity'], val_metrics['snp_precision'], 
                                 val_metrics['snp_f1_score'], "Val")
        
        # 记录INDEL指标
        logger.info(f"INDEL:")
        self._log_variant_metrics(logger, train_metrics['indel_variant_metrics'], 
                                 train_metrics['indel_sensitivity'], train_metrics['indel_precision'], 
                                 train_metrics['indel_f1_score'], "Train")
        self._log_variant_metrics(logger, val_metrics['indel_variant_metrics'], 
                                 val_metrics['indel_sensitivity'], val_metrics['indel_precision'], 
                                 val_metrics['indel_f1_score'], "Val")
    
    def _log_variant_metrics(self, logger, variant_metrics, sensitivity, precision, f1_score, prefix):
        """记录变体指标"""
        logger.info(f"{prefix}----True negative: {variant_metrics['True negative']}\t"
                   f"True positive: {variant_metrics['True positive']}\t"
                   f"False positive: {variant_metrics['False positive']}\t"
                   f"False negative: {variant_metrics['False negative']}")
        logger.info(f"{prefix}----Sensitivity: {sensitivity:.6f}\t"
                   f"Precision: {precision:.6f}\t"
                   f"F1 Score: {f1_score:.6f}")
    
    def should_save_model(self, current_val_f1, best_val_f1, current_train_f1, best_train_f1):
        """判断是否应该保存模型"""
        return (current_val_f1 > best_val_f1) or \
               (current_val_f1 == best_val_f1 and current_train_f1 > best_train_f1)
    
    def get_best_f1_scores(self):
        """获取最佳F1分数"""
        return {
            'best_val_f1_score': self.val_metrics.get('f1_score', 0),
            'best_val_f1_score_snp': self.val_metrics.get('snp_f1_score', 0),
            'best_val_f1_score_indel': self.val_metrics.get('indel_f1_score', 0),
        }