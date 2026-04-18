import time
import torch
import gc
import os
import numpy as np
from swinvar.training.training_config import TrainingConfig
from swinvar.training.data_loader_manager import DataLoaderManager
from swinvar.training.model_manager import ModelManager
from swinvar.training.metrics_calculator import MetricsCalculator
from swinvar.training.trainer import Trainer


def create_alpha(labels, num_classes):
    """创建类别权重"""
    class_counts = np.bincount(labels, minlength=num_classes)
    max_count = np.max(class_counts)
    class_weights = class_counts / max_count
    return torch.tensor(class_weights, dtype=torch.float32)


def train_model(args):
    """训练模型的主函数"""
    start_all = time.time()

    # 1. 初始化配置
    config = TrainingConfig(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 初始化数据加载器
    data_loader = DataLoaderManager(args)
    data_loader.load_data()

    try:

        # 3. 获取类别权重
        class_weights = data_loader.get_class_weights(
            config.logger if not args["checkpoint"] else None
        )

        # 4. 初始化模型管理器
        model_manager = ModelManager(args, device)
        model_manager.create_model()

        # 5. 设置微调
        param_or_groups = []
        if args["ft"]:
            model_save_path = os.path.join(
                args["output_path"], "train_moe", args["file"], args["model_save_path"]
            )
            param_or_groups =  model_manager.setup_finetuning(model_save_path)
            config.model_save_path = os.path.join(
                config.output_path, f"ft_{args['model_save_path']}"
            )

        # 6. 创建优化器和损失函数
        model_manager.create_optimizer(class_weights, param_or_groups)

        # 7. 初始化指标计算器
        metrics_calculator = MetricsCalculator()

        # 8. 初始化训练器
        trainer = Trainer(config, data_loader, model_manager, metrics_calculator)

        # 9. 加载检查点（如果存在）
        if args["checkpoint"]:
            trainer.load_checkpoint()
            model_manager.load_checkpoint(args["checkpoint_path"])

        # 10. 训练循环
        epochs = config.get_epochs()
        for epoch in range(trainer.start_epoch, epochs):
            start_time = time.time()

            # 训练一个epoch
            train_results = trainer.train_epoch(epoch, epochs)

            # 验证一个epoch
            val_results = trainer.validate_epoch(epoch, epochs)

            trainer.check_early_stopping(epoch)

            # 记录结果
            trainer.log_epoch_results(epoch, epochs, train_results, val_results)

            # 调整学习率
            trainer.adjust_learning_rate()

            # 检查早停
            if trainer.should_stop_early():
                config.logger.info(f"\n{'*'*120}")
                config.logger.info(
                    f"Early stopping at epoch: {epoch + 1}! Saving model at {trainer.best_epoch + 1} "
                    f"with Train F1 Score: {trainer.best_train_f1_score:.6f}\tVal F1 Score: {trainer.best_val_f1_score:.6f}"
                )
                config.logger.info(f"{'*'*120}")
                break

            # 检查是否到达最后一个epoch
            if epoch + 1 == epochs:
                config.logger.info(f"\n{'*'*120}")
                config.logger.info(
                    f"Saving model at epoch: {trainer.best_epoch + 1} "
                    f"with Train F1 Score: {trainer.best_train_f1_score:.6f}\tVal F1 Score: {trainer.best_val_f1_score:.6f}"
                )
                config.logger.info(f"{'*'*120}\n")

            # 保存检查点
            trainer.save_checkpoint(epoch)

            # 清理资源
            trainer.cleanup()

            time_elapsed = time.time() - start_time
            config.logger.info(f"time: {time_elapsed//60:.0f}m{time_elapsed%60:.0f}s")
            config.logger.info(f"{'-'*100}")

        # 12. 记录超参数
        # best_metrics = trainer.get_best_metrics()
        # config.log_hyperparams(
        #     model_manager.model, model_manager.optimizer, best_metrics
        # )

        # 13. 保存训练过程
        # train_history = trainer.get_train_history()
        # config.save_training_process(train_history, trainer.start_epoch + epochs - 1)

        # 14. 记录训练时间
        config.log_training_time(start_all)

    finally:
        # 15. 清理资源
        data_loader.close()
        del model_manager, data_loader, trainer
        gc.collect()
        torch.cuda.empty_cache()
