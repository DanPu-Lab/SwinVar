import tables
from torch.utils.data import DataLoader
from swinvar.models.dataset import CallingDataset
from swinvar.inference.config_call import CallConfig


class CallDataLoader:
    """测试数据加载器类"""
    
    def __init__(self, config: CallConfig):
        self.config = config
        self.tables_data_list = []
        self.dataloader = None
        self.device_config = None
        
    def setup_test_data(self):
        """设置测试数据"""
        # 获取测试输入文件
        inputs_files = self.config.get_input_files()
        
        # 打开数据文件
        self.tables_data_list = [tables.open_file(file, "r") for file in inputs_files]
        
        # 创建数据集
        dataset = CallingDataset(self.tables_data_list)
        
        # 创建数据加载器
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config.args["call_batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=self.config.args["num_workers"],
        )
        
        return self.dataloader
    
    def get_dataloader(self):
        """获取数据加载器"""
        if self.dataloader is None:
            self.setup_test_data()
        return self.dataloader
    
    def close(self):
        """关闭数据文件"""
        for tables_data in self.tables_data_list:
            tables_data.close()
    
    def __len__(self):
        """获取数据加载器长度"""
        if self.dataloader is not None:
            return len(self.dataloader)
        return 0
    
    def __iter__(self):
        """迭代器接口"""
        if self.dataloader is not None:
            return iter(self.dataloader)
        return iter([])