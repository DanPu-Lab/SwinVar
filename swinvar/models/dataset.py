import numpy as np
import torch

from torch.utils.data import Dataset


class CallingDataset(Dataset):
    def __init__(self, tables_datas):
        self.datas = tables_datas
        self.data_lengths = [len(data.root.Features) for data in tables_datas]
        self.cumulative_lengths = torch.tensor([0] + self.data_lengths).cumsum(0).tolist()

    def __len__(self):
        return sum(self.data_lengths)

    def __getitem__(self, idx):
        data_idx = next(i for i, length in enumerate(self.cumulative_lengths) if idx < length) - 1
        data = self.datas[data_idx]

        idx = idx - self.cumulative_lengths[data_idx]

        features = torch.from_numpy(data.root.Features[idx])
        variant_labels_1 = torch.from_numpy(data.root.Variant_labels[idx][0].reshape(-1)).long()
        variant_labels_2 = torch.from_numpy(data.root.Variant_labels[idx][1].reshape(-1)).long()
        genotype_labels = torch.from_numpy(data.root.Variant_labels[idx][2].reshape(-1)).long()
        chrom, pos, ref, indel_info = data.root.ChromPosRef[idx][0].decode().split(":")
        
        return features, variant_labels_1, variant_labels_2, genotype_labels, chrom, pos, ref, indel_info


if __name__ == '__main__':
    import tables
    from torch.utils.data import DataLoader

    train_tables_data_list = [tables.open_file("/data2/lijie/result/Transformer_pileup_3_channel/HG002_WES/balance/train_val_1.h5", "r")]
    
    train_dataset = CallingDataset(train_tables_data_list)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
    )

    for _ in train_dataloader:
        pass
    for tables_data in train_tables_data_list:
        tables_data.close()