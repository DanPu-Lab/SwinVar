import os
import numpy as np
import tables
import torch
from torch.utils.data import DataLoader
from swinvar.preprocess.parameters import VARIANT_SIZE, GENOTYPE_SIZE
from swinvar.models.dataset import CallingDataset


class DataLoaderManager:
    """数据加载器管理类"""
    
    def __init__(self, args):
        self.args = args
        self.train_tables_data_list = []
        self.val_tables_data_list = []
        self.train_dataloader = None
        self.val_dataloader = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup_data_paths(self):
        """设置数据路径"""
        if isinstance(self.args["input_path"], list):
            train_inputs_files = []
            val_inputs_files = []
            for input_file in self.args["input_path"]:
                input_path = os.path.join(input_file, self.args["file"]) if not self.args["ft"] else os.path.join(input_file, self.args["ft_file"])
                all_files = os.listdir(input_path)
                train_inputs_files.extend(os.path.join(input_path, file) for file in all_files if "20." not in file)
                val_inputs_files.extend(os.path.join(input_path, file) for file in all_files if "20." in file)
        else:
            input_path = os.path.join(self.args["input_path"], self.args["file"]) if not self.args["ft"] else os.path.join(self.args["input_path"], self.args["ft_file"])
            all_files = os.listdir(input_path)
            train_inputs_files = [os.path.join(input_path, file) for file in all_files if "20." not in file]
            val_inputs_files = [os.path.join(input_path, file) for file in all_files if "20." in file]
        
        return train_inputs_files, val_inputs_files
    
    def load_data(self):
        """加载数据"""
        train_inputs_files, val_inputs_files = self.setup_data_paths()
        
        self.train_tables_data_list = [tables.open_file(file, "r") for file in train_inputs_files]
        self.val_tables_data_list = [tables.open_file(file, "r") for file in val_inputs_files]
        
        train_dataset = CallingDataset(self.train_tables_data_list)
        val_dataset = CallingDataset(self.val_tables_data_list)
        
        batch_size = self.args["ft_batch_size"] if self.args["ft"] else self.args["batch_size"]
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.args["num_workers"],
        )
        
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.args["num_workers"],
        )
    
    def get_class_weights(self, logger=None):
        """获取类别权重"""
        variant_1_labels_list = []
        variant_2_labels_list = []
        genotype_labels_list = []
        
        for tables_data in self.train_tables_data_list:
            variant_1_labels_list.append(tables_data.root.Variant_labels[:][:, 0])
            variant_2_labels_list.append(tables_data.root.Variant_labels[:][:, 1])
            genotype_labels_list.append(tables_data.root.Variant_labels[:][:, 2])
        
        all_variant_1_labels = np.concatenate(variant_1_labels_list, axis=0)
        all_variant_2_labels = np.concatenate(variant_2_labels_list, axis=0)
        all_genotype_labels = np.concatenate(genotype_labels_list, axis=0)
        
        variant_1_labels_counts = np.bincount(all_variant_1_labels, minlength=VARIANT_SIZE)
        variant_2_labels_counts = np.bincount(all_variant_2_labels, minlength=VARIANT_SIZE)
        genotype_labels_counts = np.bincount(all_genotype_labels, minlength=GENOTYPE_SIZE)
        
        if logger:
            logger.info(f"Train Labels_1 Counts: {variant_1_labels_counts}")
            logger.info(f"Train Labels_2 Counts: {variant_2_labels_counts}")
            logger.info(f"Train Genotype Labels Counts: {genotype_labels_counts}")
        
        alpha_variant_1 = self._create_alpha(all_variant_1_labels, VARIANT_SIZE).to(self.device)
        alpha_variant_2 = self._create_alpha(all_variant_2_labels, VARIANT_SIZE).to(self.device)
        alpha_genotype = self._create_alpha(all_genotype_labels, GENOTYPE_SIZE).to(self.device)
        
        # 固定的权重和gamma值
        alpha_variant_1 = torch.tensor([1, 1, 1, 1, 2, 2], dtype=torch.float32, device=self.device)
        alpha_variant_2 = torch.tensor([1, 1, 1, 1, 2, 2], dtype=torch.float32, device=self.device)
        # alpha_genotype = torch.tensor([1, 2, 2, 4], dtype=torch.float32, device=self.device)
        alpha_genotype = torch.tensor([0.4, 0.4, 0.4, 2.6], dtype=torch.float32, device=self.device)
        gamma_variant_1 = torch.tensor([1, 1, 1, 1, 2, 2], dtype=torch.float32, device=self.device)
        gamma_variant_2 = torch.tensor([1, 1, 1, 1, 2, 2], dtype=torch.float32, device=self.device)
        # gamma_genotype = torch.tensor([1, 2, 2, 4], dtype=torch.float32, device=self.device)
        gamma_genotype = torch.tensor([1, 1, 1, 1], dtype=torch.float32, device=self.device)
        
        return {
            'alphas': [alpha_variant_1, alpha_variant_2, alpha_genotype],
            'gammas': [gamma_variant_1, gamma_variant_2, gamma_genotype]
        }
    
    def _create_alpha(self, labels, num_classes):
        """创建alpha权重"""
        class_counts = np.bincount(labels, minlength=num_classes)
        max_count = np.max(class_counts)
        class_weights = class_counts / max_count
        return torch.tensor(class_weights, dtype=torch.float32)
    
    def close(self):
        """关闭数据文件"""
        for tables_data in self.train_tables_data_list:
            tables_data.close()
        
        for tables_data in self.val_tables_data_list:
            tables_data.close()