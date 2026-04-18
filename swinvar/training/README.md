# SwinVar 训练和测试脚本重构说明

## 重构概述

原始的 `train.py` 和 `test.py` 文件已经按照GitHub项目结构的标准形式进行了重构，将单一大文件分解为多个模块化的类，提高了代码的可维护性和可扩展性。

## 新的项目结构

```
SwinVar/scripts/
├── __init__.py                 # 模块初始化文件
├── README.md                   # 本说明文档
├── TEST_README.md              # 测试模块详细说明
├── 
├── 训练模块
├── train.py                    # 主训练入口
├── training_config.py          # 训练配置管理类
├── data_loader_manager.py      # 数据加载器管理类
├── model_manager.py            # 模型管理类
├── metrics_calculator.py       # 指标计算类
├── trainer.py                  # 训练器类
├── 
├── 测试模块
├── test.py                     # 主测试入口
├── test_config.py              # 测试配置管理类
├── test_data_loader.py         # 测试数据加载器类
└── model_tester.py             # 模型测试器类
```

## 重构特点

### 1. 模块化设计
- 每个类都有明确的职责和功能
- 低耦合、高内聚的设计原则
- 便于独立测试和维护

### 2. 代码复用
- 测试模块充分利用训练模块的组件
- 避免重复实现相同功能
- 确保训练和测试的一致性

### 3. 可扩展性
- 易于添加新功能
- 支持不同的配置和参数
- 便于调试和问题定位

## 使用方法

### 训练模型
```python
from scripts import train_model

args = {
    "output_path": "./output",
    "file": "training_data",
    "batch_size": 32,
    "epochs": 100,
    # ... 其他参数
}
train_model(args)
```

### 测试模型
```python
from scripts import test_model

args = {
    "output_path": "./output",
    "test_input_path": "./test_data",
    "test_file": "test_samples",
    "test_batch_size": 32,
    # ... 其他参数
}
test_model(args)
```

## 主要改进

1. **代码结构**: 从单一大文件分解为多个专门的类
2. **可维护性**: 清晰的职责分离和模块化设计
3. **可读性**: 详细的注释和文档字符串
4. **错误处理**: 更好的错误隔离和处理机制
5. **资源管理**: 自动资源清理和内存管理
6. **日志记录**: 统一和改进的日志系统

## 兼容性

重构后的代码完全保持了与原始代码的兼容性：
- 相同的函数接口
- 相同的参数格式
- 相同的输出结果
- 相同的性能表现

## 详细文档

- 训练模块详细说明：参考 `TEST_README.md`
- 各模块的API文档：参考各模块的文档字符串

## 技术栈

- **Python 3.x**
- **PyTorch**
- **NumPy**
- **Pandas**
- **tables (HDF5)**
- **tqdm** (进度条)

## 贡献指南

1. 遵循现有的代码结构和命名规范
2. 添加适当的文档字符串和注释
3. 确保新功能有相应的测试
4. 更新相关文档