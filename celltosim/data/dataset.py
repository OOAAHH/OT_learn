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


class MultiModalAnnDataset(Dataset):
    """
    多模态(多样本、多时间点、多基因模块)的AnnData数据集
    
    Args:
        adata: AnnData对象
        sample_key: 样本标识列名，用于区分u₁, u₂, ..., uₙ
        time_key: 时间标签列名
        module_key: 基因模块列名(在var中)
        transport_key: 传输标签列名(区分源和目标)
        include_keys: 是否在返回数据中包含标签信息
    """
    def __init__(
        self, 
        adata, 
        sample_key=None,
        time_key=None, 
        module_key=None,
        transport_key='transport',
        include_keys=False
    ):
        self.adata = adata
        # 将AnnData中的X矩阵转为float32
        if sparse.issparse(self.adata.X):
            self.adata.X = self.adata.X.todense()
        self.adata.X = self.adata.X.astype(np.float32)
        
        # 保存标签列名
        self.sample_key = sample_key
        self.time_key = time_key
        self.module_key = module_key
        self.transport_key = transport_key
        self.include_keys = include_keys
        
        # 预处理标签信息
        self._preprocess_keys()
    
    def _preprocess_keys(self):
        """预处理和验证标签信息"""
        # 样本标签
        if self.sample_key is not None and self.sample_key in self.adata.obs:
            self.sample_categories = sorted(self.adata.obs[self.sample_key].unique())
            self.sample_to_idx = {s: i for i, s in enumerate(self.sample_categories)}
        else:
            self.sample_categories = None
            
        # 时间标签
        if self.time_key is not None and self.time_key in self.adata.obs:
            self.time_categories = sorted(self.adata.obs[self.time_key].unique())
            self.time_to_idx = {t: i for i, t in enumerate(self.time_categories)}
        else:
            self.time_categories = None
            
        # 模块标签
        if self.module_key is not None and self.module_key in self.adata.var:
            self.module_categories = sorted(self.adata.var[self.module_key].unique())
            self.module_to_idx = {m: i for i, m in enumerate(self.module_categories)}
            
            # 创建基因到模块的映射
            self.gene_to_module = {}
            for gene, module in zip(self.adata.var_names, self.adata.var[self.module_key]):
                self.gene_to_module[gene] = module
        else:
            self.module_categories = None

    def __len__(self):
        """返回数据集的样本数量"""
        return len(self.adata)

    def __getitem__(self, idx):
        """获取第idx个样本，包含可能的多模态标签信息"""
        # 基本表达值
        value = self.adata.X[idx]
        
        # 如果不需要返回标签信息，直接返回表达值
        if not self.include_keys:
            return value
        
        # 准备标签信息
        meta = {}
        
        # 样本标签
        if self.sample_key is not None and self.sample_categories is not None:
            sample = self.adata.obs[self.sample_key].iloc[idx]
            meta['sample'] = self.sample_to_idx[sample]
            
        # 时间标签
        if self.time_key is not None and self.time_categories is not None:
            time = self.adata.obs[self.time_key].iloc[idx]
            meta['time'] = self.time_to_idx[time]
            
        # 传输标签
        if self.transport_key in self.adata.obs:
            meta['transport'] = self.adata.obs[self.transport_key].iloc[idx]
        
        # 返回表达值和元信息
        return value, meta
        
    def get_metadata(self):
        """返回数据集的元数据信息"""
        return {
            'n_samples': len(self.sample_categories) if self.sample_categories else 0,
            'n_times': len(self.time_categories) if self.time_categories else 0,
            'n_modules': len(self.module_categories) if self.module_categories else 0,
            'sample_categories': self.sample_categories,
            'time_categories': self.time_categories,
            'module_categories': self.module_categories
        }
    
    def get_sample_indices(self, sample_name):
        """获取特定样本的所有索引"""
        if self.sample_key is None or sample_name not in self.sample_categories:
            return []
        
        indices = self.adata.obs[self.adata.obs[self.sample_key] == sample_name].index
        return [self.adata.obs_names.get_loc(idx) for idx in indices]
    
    def get_time_indices(self, time_name):
        """获取特定时间点的所有索引"""
        if self.time_key is None or time_name not in self.time_categories:
            return []
        
        indices = self.adata.obs[self.adata.obs[self.time_key] == time_name].index
        return [self.adata.obs_names.get_loc(idx) for idx in indices]
    
    def get_module_indices(self, module_name):
        """获取特定模块的所有基因列索引"""
        if self.module_key is None or module_name not in self.module_categories:
            return []
        
        indices = self.adata.var[self.adata.var[self.module_key] == module_name].index
        return [self.adata.var_names.get_loc(idx) for idx in indices]


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