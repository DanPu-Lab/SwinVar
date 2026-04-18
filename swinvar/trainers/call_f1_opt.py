import time

from swinvar.core.f1_threshold_trainer import F1ThresholdTrainer
from swinvar.inference.config_call import CallConfig
from swinvar.inference.data_call import CallDataLoader


def optimize_thresholds(args):
    """测试模型的主函数"""
    start_time = time.time()
    
    # 1. 初始化测试配置
    config = CallConfig(args)
    
    # 2. 初始化测试数据加载器
    data_loader = CallDataLoader(config)
    
    # 3. 初始化模型测试器
    caller = F1ThresholdTrainer(config, data_loader)
    
    # 4. 加载模型
    caller.load_model()
    
    # 5. 记录模型信息
    model_info = caller.get_model_info()
    if model_info:
        config.logger.info(f"Model loaded from: {model_info['model_path']}")
        config.logger.info(f"Total parameters: {model_info['total_parameters']:,}")
        config.logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    # 6. 运行测试
    config.logger.info("Starting model testing...")
    caller.run_test()
    
    # 8. 记录总时间
    config.log_time_taken(start_time, "[TIME TAKEN] Total testing time")
    
    # 9. 清理资源
    caller.cleanup()
