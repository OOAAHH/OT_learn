"""
数据加载函数
"""

import anndata
import numpy as np
from pathlib import Path
from collections import namedtuple

from .dataset import AnnDataDataset, split_anndata
from .utils import cast_dataset_to_loader

# 自定义命名元组，用于存储数据加载器
DataLoaders = namedtuple('DataLoaders', ['train_source', 'train_target', 'test_source', 'test_target'])


def load_anndata(path):
    """
    加载AnnData文件
    
    Args:
        path: AnnData文件路径
    Returns:
        AnnData对象
    """
    if isinstance(path, str):
        path = Path(path)
    
    assert path.exists(), f"文件不存在：{path}"
    return anndata.read(path)


def prepare_cellot_data(adata, source_label, target_label, transport_key='transport', 
                         test_size=0.2, batch_size=128, random_state=42):
    """
    准备CellOT模型的数据集和加载器
    
    Args:
        adata: AnnData对象
        source_label: 源标签的值
        target_label: 目标标签的值
        transport_key: 包含源/目标标签的列名
        test_size: 测试集比例
        batch_size: 批处理大小
        random_state: 随机种子
    Returns:
        datasets: 包含数据集的字典
        loaders: 包含数据加载器的命名元组
        input_dim: 输入维度
    """
    # 确保transport_key存在
    assert transport_key in adata.obs, f"缺少{transport_key}列在观测值中"
    
    # 分离源和目标数据
    source_data = adata[adata.obs[transport_key] == source_label].copy()
    target_data = adata[adata.obs[transport_key] == target_label].copy()
    
    # 分割训练集和测试集
    train_source, test_source = split_anndata(source_data, test_size=test_size, random_state=random_state)
    train_target, test_target = split_anndata(target_data, test_size=test_size, random_state=random_state)
    
    # 创建数据集
    datasets = {
        'train_source': AnnDataDataset(train_source),
        'train_target': AnnDataDataset(train_target),
        'test_source': AnnDataDataset(test_source),
        'test_target': AnnDataDataset(test_target)
    }
    
    # 计算所有数据集的大小
    dataset_sizes = {
        'train_source': len(datasets['train_source']),
        'train_target': len(datasets['train_target']),
        'test_source': len(datasets['test_source']),
        'test_target': len(datasets['test_target'])
    }
    
    # 计算最小训练集和测试集大小
    min_train_size = min(dataset_sizes['train_source'], dataset_sizes['train_target'])
    min_test_size = min(dataset_sizes['test_source'], dataset_sizes['test_target'])
    
    # 调整每个数据集的批次大小，确保批次大小不超过数据集大小
    train_batch_size = min(batch_size, min_train_size)
    test_batch_size = min(batch_size, min_test_size)
    
    print(f"训练集大小 - 源: {dataset_sizes['train_source']}, 目标: {dataset_sizes['train_target']}")
    print(f"测试集大小 - 源: {dataset_sizes['test_source']}, 目标: {dataset_sizes['test_target']}")
    print(f"使用的批次大小 - 训练: {train_batch_size}, 测试: {test_batch_size}")
    
    # 创建数据加载器
    loaders = DataLoaders(
        train_source=cast_dataset_to_loader(datasets['train_source'], batch_size=train_batch_size, shuffle=True),
        train_target=cast_dataset_to_loader(datasets['train_target'], batch_size=train_batch_size, shuffle=True),
        test_source=cast_dataset_to_loader(datasets['test_source'], batch_size=test_batch_size, shuffle=False),
        test_target=cast_dataset_to_loader(datasets['test_target'], batch_size=test_batch_size, shuffle=False)
    )
    
    # 获取输入维度
    input_dim = adata.n_vars
    
    return datasets, loaders, input_dim 