import torch
import time

from tqdm import tqdm
from swinvar.models.swin_var import SwinVar
from swinvar.postprocess.metrics_calculator import variant_df, calculate_metrics
from swinvar.inference.data_call import CallDataLoader
from swinvar.inference.config_call import CallConfig

class CallModel:
    """模型测试器类"""
    
    def __init__(self, config: CallConfig, data_loader: CallDataLoader):
        self.config = config
        self.data_loader = data_loader
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化测试结果存储
        self.results = {
            'variant_1_predictions': [],
            'variant_2_predictions': [],
            'genotype_predictions': [],
            'variant_1_labels': [],
            'variant_2_labels': [],
            'genotype_labels': [],
            'chrom': [],
            'pos': [],
            'ref': [],
            'indel_info': [],
        }
        
        # 统计变量
        self.nums = 0
        self.variant_1_acc = 0.0
        self.variant_2_acc = 0.0
        self.genotype_acc = 0.0
    
    def load_model(self):
        """加载模型"""
        model_config = self.config.get_model_config()
        self.model = SwinVar(**model_config).to(self.device)
        
        # 加载模型权重
        self.model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        self.model.eval()
        
        return self.model
    
    def process_batch(self, batch):
        """处理单个批次"""
        features, variant_1_labels, variant_2_labels, genotype_labels, chrom, pos, ref, indel_info = batch
        
        # 移动到设备
        features = features.to(self.device)
        variant_1_labels = variant_1_labels.reshape(-1).to(self.device)
        variant_2_labels = variant_2_labels.reshape(-1).to(self.device)
        genotype_labels = genotype_labels.reshape(-1).to(self.device)
        
        # 前向传播
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                predictions = self.model(features)
        
        return predictions, (variant_1_labels, variant_2_labels, genotype_labels, chrom, pos, ref, indel_info)
    
    def run_test(self):
        """运行测试"""
        start_time = time.time()
        
        # 确保模型已加载
        if self.model is None:
            self.load_model()
        
        # 获取数据加载器
        dataloader = self.data_loader.get_dataloader()
        
        # 测试循环
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Testing"):
                # 处理批次
                predictions, labels_data = self.process_batch(batch)
                variant_1_labels, variant_2_labels, genotype_labels, chrom, pos, ref, indel_info = labels_data
                
                # 获取预测结果
                variant_1_pred = torch.argmax(predictions[0], dim=-1)
                variant_2_pred = torch.argmax(predictions[1], dim=-1)
                genotype_pred = torch.argmax(predictions[2], dim=-1)
                
                # 存储预测结果和标签
                self.results['variant_1_predictions'].extend(variant_1_pred.cpu().numpy())
                self.results['variant_2_predictions'].extend(variant_2_pred.cpu().numpy())
                self.results['genotype_predictions'].extend(genotype_pred.cpu().numpy())
                self.results['variant_1_labels'].extend(variant_1_labels.cpu().numpy())
                self.results['variant_2_labels'].extend(variant_2_labels.cpu().numpy())
                self.results['genotype_labels'].extend(genotype_labels.cpu().numpy())
                self.results['chrom'].extend(chrom)
                self.results['pos'].extend(pos)
                self.results['ref'].extend(ref)
                self.results['indel_info'].extend(indel_info)
                
                # 计算准确率
                self.variant_1_acc += (variant_1_pred == variant_1_labels).sum().item()
                self.variant_2_acc += (variant_2_pred == variant_2_labels).sum().item()
                self.genotype_acc += (genotype_pred == genotype_labels).sum().item()
                self.nums += variant_1_labels.size(0)
        
        # 运行headwise_f1评估
        metrics = self.evaluate_headwise_f1()

        # --- 记录总时间 ---
        self.config.log_time_taken(start_time, "[TIME TAKEN] All batches processed successfully!")
        
        return self._calculate_final_results()
    
    def _calculate_final_results(self):
        """计算最终测试结果"""
        # 计算准确率
        avg_1_acc = self.variant_1_acc / self.nums
        avg_2_acc = self.variant_2_acc / self.nums
        avg_genotype_acc = self.genotype_acc / self.nums
        
        # 创建测试DataFrame
        call_df = variant_df(
            self.results['chrom'], 
            self.results['pos'], 
            self.results['ref'], 
            self.results['indel_info'],
            self.results['variant_1_predictions'], 
            self.results['variant_1_labels'],
            self.results['variant_2_predictions'], 
            self.results['variant_2_labels'],
            self.results['genotype_predictions'], 
            self.results['genotype_labels'],
            self.config.args["call_bam"], 
            self.config.args["output_vcf"], 
            self.config.args["reference"]
        )
        
        # 分离SNP和INDEL
        df_snp = call_df[~call_df["Label"].str.contains("Insert|Deletion", regex=True)]
        df_indel = call_df[call_df["Label"].str.contains("Insert|Deletion", regex=True)]
        
        # 计算统计信息
        counts = len(call_df)
        variant_counts = (call_df["Label_Genotype"] != 0).sum()
        
        # 计算指标
        variant, sensitivity, precision, f1_score = calculate_metrics(call_df)
        variant_snp, sensitivity_snp, precision_snp, f1_score_snp = calculate_metrics(df_snp)
        variant_indel, sensitivity_indel, precision_indel, f1_score_indel = calculate_metrics(df_indel)
        
        # 组织结果
        final_results = {
            'variant_1_acc': avg_1_acc,
            'variant_2_acc': avg_2_acc,
            'genotype_acc': avg_genotype_acc,
            'total_count': counts,
            'variant_count': variant_counts,
            'overall_metrics': {
                'variant': variant,
                'sensitivity': sensitivity,
                'precision': precision,
                'f1_score': f1_score
            },
            'snp_metrics': {
                'variant': variant_snp,
                'sensitivity': sensitivity_snp,
                'precision': precision_snp,
                'f1_score': f1_score_snp
            },
            'indel_metrics': {
                'variant': variant_indel,
                'sensitivity': sensitivity_indel,
                'precision': precision_indel,
                'f1_score': f1_score_indel
            },
            'test_df': call_df,
            'test_df_snp': df_snp,
            'test_df_indel': df_indel
        }
        
        return final_results
    
    def get_model_info(self):
        """获取模型信息"""
        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_path': self.config.model_path
            }
        return None
    
    def evaluate_headwise_f1(self):
        """运行headwise_f1评估并保存结果"""
        from swinvar.evaluation.headwise_f1 import evaluate_heads

        metrics = evaluate_heads(
            self.results,
            # 如果你的 genotype 索引不是 [0:0/0,1:1/0,2:1/1,3:1/2]，在这里改
            geno_index_00=0, geno_index_het=1, geno_index_hom=2, geno_index_het_alt=3,
            verbose=True
        )

        # 保存为 json
        import json, os
        output_path = self.config.args.get("output_path", ".")
        save_to = os.path.join(output_path, "headwise_f1.json")
        with open(save_to, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Saved] {save_to}")

        return metrics

    def cleanup(self):
        """清理资源"""
        self.data_loader.close()
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()