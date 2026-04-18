import os, torch, numpy as np
import time

from tqdm import tqdm

from swinvar.postprocess.f1_opt import GenotypeF1Optimizer
from swinvar.inference.model_call import CallModel



class F1ThresholdTrainer(CallModel):
    def run_test(self, json_path="configs"):
        """运行测试"""
        start_time = time.time()
        
        # 确保模型已加载
        if self.model is None:
            self.load_model()
        
        # 获取数据加载器
        dataloader = self.data_loader.get_dataloader()

        # 这些需要根据你的数据统计替换（训练集先验 & 真实部署先验）
        TRAIN_PRIORS = [0.666666667, 0.198079575, 0.134747329, 0.000506429]   # 示例：按你训练集的 {0/0,1/0,1/1,1/2} 占比替换
        TEST_PRIORS  = [0.989837399, 0.006039011, 0.00410815, 0.00001544]  # 示例：100:1 的真实分布替换

        # 在 val 循环里缓存
        val_geno_logits = []
        val_geno_labels = []

        # 测试循环
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Call with PR thresholds"):
                # 处理批次
                predictions, labels_data = self.process_batch(batch)
                
                genotype_pred = predictions[2]
                genotype_labels = labels_data[2]
                val_geno_logits.append(genotype_pred.detach().cpu())
                val_geno_labels.append(genotype_labels.detach().cpu())

        val_geno_logits = torch.cat(val_geno_logits, dim=0)  # (Nv,4)
        val_geno_labels = torch.cat(val_geno_labels, dim=0)  # (Nv,)

        # 拟合 & 保存
        opt = GenotypeF1Optimizer(train_priors=TRAIN_PRIORS, test_priors=TEST_PRIORS, geno_index_00=0)
        opt.fit(val_geno_logits, val_geno_labels)

        os.makedirs(json_path, exist_ok=True)
        torch.save(opt.state_dict(), os.path.join(json_path, "geno_f1_opt.pt"))
                
        # 记录总时间
        self.config.log_time_taken(start_time, "[TIME TAKEN] All batches processed successfully!")
        
