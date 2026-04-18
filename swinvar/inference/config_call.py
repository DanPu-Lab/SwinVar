import os
import time
from swinvar.preprocess.utils import setup_logger, check_directory


class CallConfig:
    """测试配置管理类"""
    
    def __init__(self, args):
        self.args = args
        self._setup_paths()
        self._setup_logging()
        self._validate_config()
    
    def _setup_paths(self):
        """设置测试路径配置"""
        if self.args["ft"]:
            self.output_path = os.path.join(
                self.args["output_path"], 
                "train_moe", 
                f"ft_{self.args['ft_file']}_{self.args['ref_var_ratio']}"
            )
            self.model_path = os.path.join(self.output_path, f"ft_{self.args['model_save_path']}")
        else:
            self.output_path = os.path.join(self.args["output_path"], "train_moe", f"{self.args['file']}")
            self.model_path = os.path.join(self.args["output_path"], "train_moe", f"{self.args['file']}", f"{self.args['model_save_path']}")
        
        check_directory(self.output_path)
        
        self.log_file = os.path.join(self.output_path, self.args["log_file_call"])
        self.input_path = os.path.join(self.args["call_input_path"], self.args["call_file"])
    
    def _setup_logging(self):
        """设置日志"""
        self.logger = setup_logger(self.log_file)
    
    def _validate_config(self):
        """验证配置"""
        required_keys = [
            "call_batch_size", "num_workers", "feature_size", "num_classes",
            "embed_dim", "patch_size", "window_size", "n_routed_experts",
            "n_activated_experts", "n_expert_groups", "n_limited_groups",
            "score_func", "route_scale", "moe_inter_dim", "n_shared_experts",
            "depths", "num_heads", "drop_rate", "drop_path_rate", "attn_drop_rate",
            "output_vcf", "reference"
        ]
        
        for key in required_keys:
            if key not in self.args:
                raise ValueError(f"Missing required test config key: {key}")
    
    def get_input_files(self):
        """获取测试输入文件列表"""
        return [os.path.join(self.input_path, file) for file in os.listdir(self.input_path)]
    
    def log_time_taken(self, start_time, message_prefix="[TIME TAKEN]"):
        """记录时间消耗"""
        consum_time = time.time() - start_time
        hours, remainder = divmod(consum_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{hours:.0f}h{minutes:.0f}m{seconds:.0f}s"
        self.logger.info(f"{message_prefix} Total time: {time_str}")
        return time_str
    
    def log_test_results(self, test_results):
        """记录测试结果"""
        logger = setup_logger(self.log_file, mode="a")
        
        # 记录准确率
        logger.info(
            f"测试集ACC: Genotype: {test_results['genotype_acc']:.6f}\t"
            f"Variant 1: {test_results['variant_1_acc']:.6f}\t"
            f"Variant 2: {test_results['variant_2_acc']:.6f}"
        )
        
        # 记录总体统计
        logger.info(f"Total: {test_results['total_count']}")
        logger.info(f"Total number of variants: {test_results['variant_count']}")
        
        # 记录整体指标
        self._log_variant_metrics(
            logger, test_results['overall_metrics'], 
            "Overall"
        )
        
        # 记录SNP指标
        logger.info(f"SNP:")
        self._log_variant_metrics(
            logger, test_results['snp_metrics'], 
            "SNP"
        )
        
        # 记录INDEL指标
        logger.info(f"INDEL:")
        self._log_variant_metrics(
            logger, test_results['indel_metrics'], 
            "INDEL"
        )
    
    def _log_variant_metrics(self, logger, metrics, prefix):
        """记录变体指标"""
        logger.info(
            f"True negative: {metrics['variant']['True negative']}\t"
            f"True positive: {metrics['variant']['True positive']}\t"
            f"False positive: {metrics['variant']['False positive']}\t"
            f"False negative: {metrics['variant']['False negative']}"
        )
        logger.info(
            f"Sensitivity: {metrics['sensitivity']:.6f}\t"
            f"Precision: {metrics['precision']:.6f}\t"
            f"F1 score: {metrics['f1_score']:.6f}"
        )
    
    def get_model_config(self):
        """获取模型配置参数"""
        return {
            'feature_size': self.args['feature_size'],
            'num_classes': self.args['num_classes'],
            'embed_dim': self.args['embed_dim'],
            'patch_size': self.args['patch_size'],
            'window_size': self.args['window_size'],
            'n_routed_experts': self.args['n_routed_experts'],
            'n_activated_experts': self.args['n_activated_experts'],
            'n_expert_groups': self.args['n_expert_groups'],
            'n_limited_groups': self.args['n_limited_groups'],
            'score_func': self.args['score_func'],
            'route_scale': self.args['route_scale'],
            'moe_inter_dim': self.args['moe_inter_dim'],
            'n_shared_experts': self.args['n_shared_experts'],
            'depths': self.args['depths'],
            'num_heads': self.args['num_heads'],
            'drop_rate': self.args['drop_rate'],
            'drop_path_rate': self.args['drop_path_rate'],
            'attn_drop_rate': self.args['attn_drop_rate'],
        }