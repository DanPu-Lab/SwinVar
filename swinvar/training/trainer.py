from os import system
import time
import torch
import gc
from tqdm import tqdm

from swinvar.training.model_manager import ModelManager
from swinvar.postprocess.metrics_calculator import variant_df, calculate_metrics
from swinvar.training.metrics_calculator import MetricsCalculator


class Trainer:
    """训练器类"""

    def __init__(
        self,
        config,
        data_loader,
        model_manager: ModelManager,
        metrics_calculator: MetricsCalculator,
    ):
        self.config = config
        self.data_loader = data_loader
        self.model_manager = model_manager
        self.metrics_calculator = metrics_calculator

        # 初始化训练状态
        self.best_val_loss = float("inf")
        self.best_val_acc = float("-inf")
        self.best_train_f1_score = float("-inf")
        self.best_train_f1_score_indel = float("-inf")
        self.best_val_f1_score = float("-inf")
        self.best_val_f1_score_snp = float("-inf")
        self.best_val_f1_score_indel = float("-inf")
        self.best_epoch = 0

        self.patience_counter = 0
        self.patience = config.get_patience()
        self.patience_lr_scheduler = self.patience // 4

        # 训练历史记录
        self.train_history = {
            "train_loss_list": [],
            "val_loss_list": [],
            "train_variant_1_acc_list": [],
            "val_variant_1_acc_list": [],
            "train_variant_2_acc_list": [],
            "val_variant_2_acc_list": [],
            "train_genotype_acc_list": [],
            "val_genotype_acc_list": [],
            "train_sensitivity_list": [],
            "val_sensitivity_list": [],
            "train_precision_list": [],
            "val_precision_list": [],
            "train_f1_score_list": [],
            "val_f1_score_list": [],
            "train_f1_score_snp_list": [],
            "val_f1_score_snp_list": [],
            "train_f1_score_indel_list": [],
            "val_f1_score_indel_list": [],
        }

        self.start_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_epoch(self, epoch, total_epochs):
        """训练一个epoch"""
        self.model_manager.train()
        epoch_start_time = time.time()

        # 初始化统计变量
        train_loss = 0.0
        train_variant_1_acc = 0.0
        train_variant_2_acc = 0.0
        train_genotype_acc = 0.0
        train_num = 0

        # 收集预测结果
        train_variant_1_predictions = []
        train_variant_2_predictions = []
        train_genotype_predictions = []
        train_variant_1_labels = []
        train_variant_2_labels = []
        train_genotype_labels = []
        train_chrom = []
        train_pos = []
        train_ref = []
        train_indel_info = []

        # 训练循环
        for batch in tqdm(
            self.data_loader.train_dataloader, desc=f"Training Epoch {epoch+1}"
        ):
            (
                features,
                variant_1_labels,
                variant_2_labels,
                genotype_labels,
                chrom,
                pos,
                ref,
                indel_info,
            ) = batch

            # 移动到设备
            features = features.to(self.device)
            variant_1_labels = variant_1_labels.reshape(-1).to(self.device)
            variant_2_labels = variant_2_labels.reshape(-1).to(self.device)
            genotype_labels = genotype_labels.reshape(-1).to(self.device)

            # 清零梯度
            self.model_manager.zero_grad()

            # 前向传播
            with torch.amp.autocast("cuda"):
                (
                    variant_1_classification,
                    variant_2_classification,
                    genotype_classification,
                ) = self.model_manager.forward(features)
                loss = self.model_manager.criterion(
                    [
                        variant_1_classification,
                        variant_2_classification,
                        genotype_classification,
                    ],
                    [variant_1_labels, variant_2_labels, genotype_labels],
                )

            # 反向传播
            scaled_loss = self.model_manager.scale_loss(loss)
            self.model_manager.backward(scaled_loss)
            self.model_manager.step()
            self.model_manager.update_scaler()

            # 计算准确率
            train_variant_1_pred = torch.argmax(variant_1_classification, dim=-1)
            train_variant_2_pred = torch.argmax(variant_2_classification, dim=-1)
            train_genotype_pred = torch.argmax(genotype_classification, dim=-1)

            train_variant_1_predictions.extend(train_variant_1_pred.cpu().numpy())
            train_variant_2_predictions.extend(train_variant_2_pred.cpu().numpy())
            train_genotype_predictions.extend(train_genotype_pred.cpu().numpy())
            train_variant_1_labels.extend(variant_1_labels.cpu().numpy())
            train_variant_2_labels.extend(variant_2_labels.cpu().numpy())
            train_genotype_labels.extend(genotype_labels.cpu().numpy())
            train_chrom.extend(chrom)
            train_pos.extend(pos)
            train_ref.extend(ref)
            train_indel_info.extend(indel_info)

            train_variant_1_acc += (
                (train_variant_1_pred == variant_1_labels).sum().item()
            )
            train_variant_2_acc += (
                (train_variant_2_pred == variant_2_labels).sum().item()
            )
            train_genotype_acc += (train_genotype_pred == genotype_labels).sum().item()
            train_loss += loss.item() * features.size(0)
            train_num += features.size(0)

        # 计算平均损失和准确率
        self.train_history["train_loss_list"].append(train_loss / train_num)
        self.train_history["train_variant_1_acc_list"].append(
            train_variant_1_acc / train_num
        )
        self.train_history["train_variant_2_acc_list"].append(
            train_variant_2_acc / train_num
        )
        self.train_history["train_genotype_acc_list"].append(
            train_genotype_acc / train_num
        )

        train_df = variant_df(
            train_chrom,
            train_pos,
            train_ref,
            train_indel_info,
            train_variant_1_predictions,
            train_variant_1_labels,
            train_variant_2_predictions,
            train_variant_2_labels,
            train_genotype_predictions,
            train_genotype_labels,
        )
        train_df_snp = train_df[
            ~train_df["Label"].str.contains("Insert|Deletion", regex=True)
        ]
        train_df_indel = train_df[
            train_df["Label"].str.contains("Insert|Deletion", regex=True)
        ]

        train_variant_count = (train_df["Label_Genotype"] != 0).sum()
        train_variant, train_sensitivity, train_precision, train_f1_score = (
            calculate_metrics(train_df)
        )

        self.train_history["train_sensitivity_list"].append(train_sensitivity)
        self.train_history["train_precision_list"].append(train_precision)
        self.train_history["train_f1_score_list"].append(train_f1_score)

        # SNP
        (
            train_variant_snp,
            train_sensitivity_snp,
            train_precision_snp,
            train_f1_score_snp,
        ) = calculate_metrics(train_df_snp)
        self.train_history["train_f1_score_snp_list"].append(train_f1_score_snp)

        # INDEL
        (
            train_variant_indel,
            train_sensitivity_indel,
            train_precision_indel,
            train_f1_score_indel,
        ) = calculate_metrics(train_df_indel)
        self.train_history["train_f1_score_indel_list"].append(train_f1_score_indel)

        return {
            "loss": self.train_history["train_loss_list"][-1],
            "genotype_acc": self.train_history["train_genotype_acc_list"][-1],
            "variant_1_acc": self.train_history["train_variant_1_acc_list"][-1],
            "variant_2_acc": self.train_history["train_variant_2_acc_list"][-1],
            "variant_count": train_variant_count,
            "variant_metrics": train_variant,
            "sensitivity": train_sensitivity,
            "precision": train_precision,
            "f1_score": train_f1_score,
            "snp_variant_metrics": train_variant_snp,
            "snp_sensitivity": train_sensitivity_snp,
            "snp_precision": train_precision_snp,
            "snp_f1_score": train_f1_score_snp,
            "indel_variant_metrics": train_variant_indel,
            "indel_sensitivity": train_sensitivity_indel,
            "indel_precision": train_precision_indel,
            "indel_f1_score": train_f1_score_indel,
            "time_elapsed": time.time() - epoch_start_time,
        }

    def validate_epoch(self, epoch, total_epochs):
        """验证一个epoch"""
        self.model_manager.eval()
        epoch_start_time = time.time()

        # 初始化统计变量
        val_loss = 0.0
        val_variant_1_acc = 0.0
        val_variant_2_acc = 0.0
        val_genotype_acc = 0.0
        val_num = 0

        # 收集预测结果
        val_variant_1_predictions = []
        val_variant_2_predictions = []
        val_genotype_predictions = []
        val_variant_1_labels = []
        val_variant_2_labels = []
        val_genotype_labels = []
        val_chrom = []
        val_pos = []
        val_ref = []
        val_indel_info = []

        # 验证循环
        with torch.no_grad():
            for batch in tqdm(
                self.data_loader.val_dataloader, desc=f"Validating Epoch {epoch+1}"
            ):
                (
                    features,
                    variant_1_labels,
                    variant_2_labels,
                    genotype_labels,
                    chrom,
                    pos,
                    ref,
                    indel_info,
                ) = batch

                # 移动到设备
                features = features.to(self.device)
                variant_1_labels = variant_1_labels.reshape(-1).to(self.device)
                variant_2_labels = variant_2_labels.reshape(-1).to(self.device)
                genotype_labels = genotype_labels.reshape(-1).to(self.device)

                # 前向传播
                with torch.amp.autocast("cuda"):
                    (
                        variant_1_classification,
                        variant_2_classification,
                        genotype_classification,
                    ) = self.model_manager.forward(features)
                    loss = self.model_manager.criterion(
                        [
                            variant_1_classification,
                            variant_2_classification,
                            genotype_classification,
                        ],
                        [variant_1_labels, variant_2_labels, genotype_labels],
                    )

                # 计算准确率
                val_variant_1_pred = torch.argmax(variant_1_classification, dim=-1)
                val_variant_2_pred = torch.argmax(variant_2_classification, dim=-1)
                val_genotype_pred = torch.argmax(genotype_classification, dim=-1)

                val_variant_1_predictions.extend(val_variant_1_pred.cpu().numpy())
                val_variant_2_predictions.extend(val_variant_2_pred.cpu().numpy())
                val_genotype_predictions.extend(val_genotype_pred.cpu().numpy())
                val_variant_1_labels.extend(variant_1_labels.cpu().numpy())
                val_variant_2_labels.extend(variant_2_labels.cpu().numpy())
                val_genotype_labels.extend(genotype_labels.cpu().numpy())
                val_chrom.extend(chrom)
                val_pos.extend(pos)
                val_ref.extend(ref)
                val_indel_info.extend(indel_info)

                val_variant_1_acc += (
                    (val_variant_1_pred.detach() == variant_1_labels.detach())
                    .sum()
                    .item()
                )
                val_variant_2_acc += (
                    (val_variant_2_pred.detach() == variant_2_labels.detach())
                    .sum()
                    .item()
                )
                val_genotype_acc += (
                    (val_genotype_pred.detach() == genotype_labels.detach())
                    .sum()
                    .item()
                )
                val_loss += loss.item() * features.size(0)
                val_num += features.size(0)

        # 计算平均损失和准确率
        self.train_history["val_loss_list"].append(val_loss / val_num)
        self.train_history["val_variant_1_acc_list"].append(val_variant_1_acc / val_num)
        self.train_history["val_variant_2_acc_list"].append(val_variant_2_acc / val_num)
        self.train_history["val_genotype_acc_list"].append(val_genotype_acc / val_num)

        val_df = variant_df(
            val_chrom,
            val_pos,
            val_ref,
            val_indel_info,
            val_variant_1_predictions,
            val_variant_1_labels,
            val_variant_2_predictions,
            val_variant_2_labels,
            val_genotype_predictions,
            val_genotype_labels,
        )
        val_df_snp = val_df[
            ~val_df["Label"].str.contains("Insert|Deletion", regex=True)
        ]
        val_df_indel = val_df[
            val_df["Label"].str.contains("Insert|Deletion", regex=True)
        ]

        val_variant_count = (val_df["Label_Genotype"] != 0).sum()
        val_variant, val_sensitivity, val_precision, val_f1_score = calculate_metrics(
            val_df
        )

        self.train_history["val_sensitivity_list"].append(val_sensitivity)
        self.train_history["val_precision_list"].append(val_precision)
        self.train_history["val_f1_score_list"].append(val_f1_score)

        # SNP
        val_variant_snp, val_sensitivity_snp, val_precision_snp, val_f1_score_snp = (
            calculate_metrics(val_df_snp)
        )
        self.train_history["val_f1_score_snp_list"].append(val_f1_score_snp)

        # INDEL
        (
            val_variant_indel,
            val_sensitivity_indel,
            val_precision_indel,
            val_f1_score_indel,
        ) = calculate_metrics(val_df_indel)
        self.train_history["val_f1_score_indel_list"].append(val_f1_score_indel)

        return {
            "loss": self.train_history["val_loss_list"][-1],
            "genotype_acc": self.train_history["val_genotype_acc_list"][-1],
            "variant_1_acc": self.train_history["val_variant_1_acc_list"][-1],
            "variant_2_acc": self.train_history["val_variant_2_acc_list"][-1],
            "variant_count": val_variant_count,
            "variant_metrics": val_variant,
            "sensitivity": val_sensitivity,
            "precision": val_precision,
            "f1_score": val_f1_score,
            "snp_variant_metrics": val_variant_snp,
            "snp_sensitivity": val_sensitivity_snp,
            "snp_precision": val_precision_snp,
            "snp_f1_score": val_f1_score_snp,
            "indel_variant_metrics": val_variant_indel,
            "indel_sensitivity": val_sensitivity_indel,
            "indel_precision": val_precision_indel,
            "indel_f1_score": val_f1_score_indel,
            "time_elapsed": time.time() - epoch_start_time,
        }

    def check_early_stopping(self, epoch):
        """检查早停条件"""
        if (self.train_history["val_f1_score_list"][-1] > self.best_val_f1_score) or (
            self.train_history["val_f1_score_list"][-1] == self.best_val_f1_score
            and self.train_history["train_f1_score_list"][-1] > self.best_train_f1_score
        ):
            # 更新最佳指标
            self.best_train_f1_score = self.train_history["train_f1_score_list"][-1]
            self.best_val_f1_score = self.train_history["val_f1_score_list"][-1]
            self.best_epoch = epoch
            self.patience_counter = 0
            self.patience_lr_scheduler = self.patience // 4

            # 保存模型
            self.model_manager.save_model(self.config.model_save_path)
            return True
        else:
            self.patience_counter += 1
            return False

    def adjust_learning_rate(self):
        """调整学习率"""
        if (
            self.patience_counter > 0
            and self.patience_counter % self.patience_lr_scheduler == 0
        ):
            self.patience_lr_scheduler += self.patience_lr_scheduler

            for param_group in self.model_manager.optimizer.optimizer.param_groups:
                param_group["lr"] *= 0.1

    def should_stop_early(self):
        """判断是否应该早停"""
        return self.patience_counter >= self.patience

    def log_epoch_results(self, epoch, total_epochs, train_results, val_results):
        """记录epoch结果"""

        self.metrics_calculator.log_metrics(
            self.config.logger,
            epoch,
            total_epochs,
            train_results,
            val_results,
            self.patience_counter,
        )

    def save_checkpoint(self, epoch):
        """保存检查点"""
        checkpoint_dict = {
            "epoch": epoch + 1,
            "patience_counter": self.patience_counter,
            "best_val_f1_score": self.best_val_f1_score,
            "best_train_f1_score": self.best_train_f1_score,
            "best_epoch": self.best_epoch,
            "patience_lr_scheduler": self.patience_lr_scheduler,
            "train_loss_list": self.train_history["train_loss_list"],
            "val_loss_list": self.train_history["val_loss_list"],
            "train_sensitivity_list": self.train_history["train_sensitivity_list"],
            "val_sensitivity_list": self.train_history["val_sensitivity_list"],
            "train_precision_list": self.train_history["train_precision_list"],
            "val_precision_list": self.train_history["val_precision_list"],
            "train_f1_score_list": self.train_history["train_f1_score_list"],
            "val_f1_score_list": self.train_history["val_f1_score_list"],
        }

        self.model_manager.save_checkpoint(checkpoint_dict, self.config.checkpoint_path)

    def load_checkpoint(self):
        """加载检查点"""
        checkpoint_dict = self.model_manager.load_checkpoint(
            self.config.checkpoint_path
        )

        self.start_epoch = checkpoint_dict["epoch"]
        self.patience_counter = checkpoint_dict["patience_counter"]
        self.best_train_f1_score = checkpoint_dict["best_train_f1_score"]
        self.best_val_f1_score = checkpoint_dict["best_val_f1_score"]
        self.best_epoch = checkpoint_dict["best_epoch"]
        self.patience_lr_scheduler = checkpoint_dict["patience_lr_scheduler"]

        self.train_history["train_loss_list"] = checkpoint_dict["train_loss_list"]
        self.train_history["val_loss_list"] = checkpoint_dict["val_loss_list"]
        self.train_history["train_sensitivity_list"] = checkpoint_dict[
            "train_sensitivity_list"
        ]
        self.train_history["val_sensitivity_list"] = checkpoint_dict[
            "val_sensitivity_list"
        ]
        self.train_history["train_precision_list"] = checkpoint_dict[
            "train_precision_list"
        ]
        self.train_history["val_precision_list"] = checkpoint_dict["val_precision_list"]
        self.train_history["train_f1_score_list"] = checkpoint_dict[
            "train_f1_score_list"
        ]
        self.train_history["val_f1_score_list"] = checkpoint_dict["val_f1_score_list"]

        return checkpoint_dict

    def cleanup(self):
        """清理资源"""
        torch.cuda.empty_cache()
        gc.collect()

    def get_best_metrics(self):
        """获取最佳指标"""
        return {
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_val_f1_score": self.best_val_f1_score,
            "best_val_f1_score_snp": self.best_val_f1_score_snp,
            "best_val_f1_score_indel": self.best_val_f1_score_indel,
            "best_epoch": self.best_epoch,
        }

    def get_train_history(self):
        """获取训练历史"""
        return self.train_history
