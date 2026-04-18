import os, torch
import time

from tqdm import tqdm

from swinvar.postprocess.f1_opt import GenotypeF1Optimizer
from swinvar.inference.model_call import CallModel


class F1GenotypePredictor(CallModel):
    def run_test(self, json_path="configs"):
        """运行测试"""
        start_time = time.time()
        
        # 确保模型已加载
        if self.model is None:
            self.load_model()
        
        # 获取数据加载器
        dataloader = self.data_loader.get_dataloader()

        opt_sd_path = os.path.join(json_path, "geno_f1_opt.pt")
        assert os.path.exists(opt_sd_path), f"Missing {opt_sd_path} (please run validation fitting first)."
        opt = GenotypeF1Optimizer(train_priors=[0,0,0,0], test_priors=[0,0,0,0], geno_index_00=0)  # 先占位
        opt.load_state_dict(torch.load(opt_sd_path, map_location="cpu", weights_only=False))
        opt.temp = opt.temp.to(self.device)

        # 测试循环
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Testing"):
                # 处理批次
                predictions, labels_data = self.process_batch(batch)
                variant_1_labels, variant_2_labels, genotype_labels, chrom, pos, ref, indel_info = labels_data
                
                # 获取预测结果
                variant_1_pred = torch.argmax(predictions[0], dim=-1)
                variant_2_pred = torch.argmax(predictions[1], dim=-1)
                genotype_pred = predictions[2]
                _, genotype_pred_final = opt.predict(genotype_pred)
                genotype_pred = torch.from_numpy(genotype_pred_final).to(genotype_labels.device)
                
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