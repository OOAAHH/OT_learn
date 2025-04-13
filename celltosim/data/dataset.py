"""
单细胞数据集类
"""

import anndata
import numpy as np
import pandas as pd
from scipy import sparse
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class AnnDataDataset(Dataset):
    """
    将AnnData对象包装为PyTorch数据集
    
    Args:
        adata: AnnData对象
        obs_key: 观测标签列名
        categories: 类别标签列表
        include_index: 是否包含索引
    """
    def __init__(
        self, adata, obs_key=None, categories=None, include_index=False
    ):
        self.adata = adata
        # 将AnnData中的X矩阵转为float32
        if sparse.issparse(self.adata.X):
            self.adata.X = self.adata.X.todense()
        self.adata.X = self.adata.X.astype(np.float32)
        self.obs_key = obs_key
        self.categories = categories
        self.include_index = include_index

    def __len__(self):
        """返回数据集的样本数量"""
        return len(self.adata)

    def __getitem__(self, idx):
        """获取第idx个样本"""
        value = self.adata.X[idx]

        # 如果指定了标签列，返回带标签的样本
        if self.obs_key is not None and self.categories is not None:
            meta = self.categories.index(self.adata.obs[self.obs_key].iloc[idx])
            value = value, int(meta)

        # 如果需要返回索引
        if self.include_index:
            return self.adata.obs_names[idx], value

        return value


def split_anndata(adata, test_size=0.2, random_state=42, condition_key=None):
    """
    将AnnData对象分为训练集和测试集
    
    Args:
        adata: AnnData对象
        test_size: 测试集大小比例
        random_state: 随机种子
        condition_key: 分层抽样的条件键
    Returns:
        train_data, test_data: 训练集和测试集的AnnData对象
    """
    # 获取所有样本的索引
    indices = np.arange(len(adata))
    
    # 如果指定了条件键，则进行分层抽样
    if condition_key is not None and condition_key in adata.obs:
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=test_size, 
            random_state=random_state,
            stratify=adata.obs[condition_key]
        )
    else:
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=test_size, 
            random_state=random_state
        )
    
    # 创建训练集和测试集
    train_data = adata[train_idx].copy()
    test_data = adata[test_idx].copy()
    
    return train_data, test_data 