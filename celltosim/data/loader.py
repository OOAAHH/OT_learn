"""
数据加载函数
"""

import anndata
import numpy as np
from pathlib import Path
from collections import namedtuple

from .dataset import AnnDataDataset, MultiModalAnnDataset, split_anndata
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


def prepare_multimodal_data(
    species_u_adata, 
    species_v_adata, 
    transport_key='transport',
    source_label='source',
    target_label='target',
    sample_key=None, 
    time_key=None, 
    module_key=None,
    test_size=0.2, 
    batch_size=128, 
    random_state=42,
    include_keys=True
):
    """
    准备多模态数据集（多样本、多时间点、多基因模块）
    
    Args:
        species_u_adata: 物种U的AnnData对象
        species_v_adata: 物种V的AnnData对象
        transport_key: 传输标签列名
        source_label: 源标签值
        target_label: 目标标签值
        sample_key: 样本标签列名
        time_key: 时间标签列名
        module_key: 模块标签列名(在var中)
        test_size: 测试集比例
        batch_size: 批处理大小
        random_state: 随机种子
        include_keys: 是否在数据中包含标签信息
    Returns:
        datasets: 包含数据集的字典
        loaders: 包含数据加载器的命名元组
        metadata: 包含元数据信息的字典
    """
    # 确保传输标签存在于数据集中
    for adata, species in [(species_u_adata, 'U'), (species_v_adata, 'V')]:
        assert transport_key in adata.obs, f"物种{species}缺少{transport_key}列"
    
    # 分离源和目标数据
    u_source = species_u_adata[species_u_adata.obs[transport_key] == source_label].copy()
    v_target = species_v_adata[species_v_adata.obs[transport_key] == target_label].copy()
    
    # 分割训练集和测试集
    u_source_train, u_source_test = split_anndata(
        u_source, 
        test_size=test_size, 
        random_state=random_state,
        condition_key=sample_key if sample_key in u_source.obs else None
    )
    
    v_target_train, v_target_test = split_anndata(
        v_target, 
        test_size=test_size, 
        random_state=random_state,
        condition_key=sample_key if sample_key in v_target.obs else None
    )
    
    # 创建多模态数据集
    datasets = {
        'train_source': MultiModalAnnDataset(
            u_source_train, 
            sample_key=sample_key,
            time_key=time_key, 
            module_key=module_key,
            transport_key=transport_key,
            include_keys=include_keys
        ),
        'train_target': MultiModalAnnDataset(
            v_target_train, 
            sample_key=sample_key,
            time_key=time_key, 
            module_key=module_key,
            transport_key=transport_key,
            include_keys=include_keys
        ),
        'test_source': MultiModalAnnDataset(
            u_source_test, 
            sample_key=sample_key,
            time_key=time_key, 
            module_key=module_key,
            transport_key=transport_key,
            include_keys=include_keys
        ),
        'test_target': MultiModalAnnDataset(
            v_target_test, 
            sample_key=sample_key,
            time_key=time_key, 
            module_key=module_key,
            transport_key=transport_key,
            include_keys=include_keys
        )
    }
    
    # 获取数据集大小
    dataset_sizes = {k: len(v) for k, v in datasets.items()}
    
    # 计算最小训练集和测试集大小
    min_train_size = min(dataset_sizes['train_source'], dataset_sizes['train_target'])
    min_test_size = min(dataset_sizes['test_source'], dataset_sizes['test_target'])
    
    # 调整批次大小
    train_batch_size = min(batch_size, min_train_size)
    test_batch_size = min(batch_size, min_test_size)
    
    print(f"训练集大小 - 源(U): {dataset_sizes['train_source']}, 目标(V): {dataset_sizes['train_target']}")
    print(f"测试集大小 - 源(U): {dataset_sizes['test_source']}, 目标(V): {dataset_sizes['test_target']}")
    print(f"使用的批次大小 - 训练: {train_batch_size}, 测试: {test_batch_size}")
    
    # 创建数据加载器
    loaders = DataLoaders(
        train_source=cast_dataset_to_loader(datasets['train_source'], batch_size=train_batch_size, shuffle=True),
        train_target=cast_dataset_to_loader(datasets['train_target'], batch_size=train_batch_size, shuffle=True),
        test_source=cast_dataset_to_loader(datasets['test_source'], batch_size=test_batch_size, shuffle=False),
        test_target=cast_dataset_to_loader(datasets['test_target'], batch_size=test_batch_size, shuffle=False)
    )
    
    # 获取元数据
    metadata = {
        'input_dim': species_u_adata.n_vars,
        'u_metadata': datasets['train_source'].get_metadata(),
        'v_metadata': datasets['train_target'].get_metadata(),
        'sample_key': sample_key,
        'time_key': time_key,
        'module_key': module_key
    }
    
    return datasets, loaders, metadata


def merge_species_data(species_u_adata, species_v_adata, 
                      homologous_genes=None, 
                      module_key=None,
                      source_label='source', 
                      target_label='target',
                      transport_key='transport'):
    """
    合并两个物种的数据，仅保留同源基因或模块
    
    Args:
        species_u_adata: 物种U的AnnData对象
        species_v_adata: 物种V的AnnData对象
        homologous_genes: 同源基因列表(如果为None，则使用module_key)
        module_key: 基因模块标签，用于确定同源模块
        source_label: 源标签值
        target_label: 目标标签值
        transport_key: 传输标签列名
    Returns:
        merged_adata: 合并后的AnnData对象
    """
    # 确保传输标签存在于数据集中
    for adata, species in [(species_u_adata, 'U'), (species_v_adata, 'V')]:
        assert transport_key in adata.obs, f"物种{species}缺少{transport_key}列"
    
    # 复制数据以避免修改原始数据
    u_data = species_u_adata.copy()
    v_data = species_v_adata.copy()
    
    # 设置传输标签
    u_data.obs[transport_key] = source_label
    v_data.obs[transport_key] = target_label
    
    # 对齐基因(两种方式)
    if homologous_genes is not None:
        # 使用提供的同源基因列表
        u_genes = [g for g in homologous_genes if g in u_data.var_names]
        v_genes = [g for g in homologous_genes if g in v_data.var_names]
        
        # 取交集
        common_genes = list(set(u_genes) & set(v_genes))
        print(f"找到{len(common_genes)}个共同同源基因")
        
        # 仅保留共同基因
        u_data = u_data[:, common_genes].copy()
        v_data = v_data[:, common_genes].copy()
    
    elif module_key is not None and module_key in u_data.var and module_key in v_data.var:
        # 找到两个物种共有的模块
        u_modules = set(u_data.var[module_key])
        v_modules = set(v_data.var[module_key])
        common_modules = list(u_modules & v_modules)
        
        print(f"找到{len(common_modules)}个共同模块")
        
        if len(common_modules) == 0:
            raise ValueError("未找到共同模块，无法合并数据")
        
        # 对每个共同模块
        all_u_genes = []
        all_v_genes = []
        
        for module in common_modules:
            # 获取属于该模块的基因
            u_module_genes = u_data.var_names[u_data.var[module_key] == module]
            v_module_genes = v_data.var_names[v_data.var[module_key] == module]
            
            print(f"模块{module}: U物种有{len(u_module_genes)}个基因，V物种有{len(v_module_genes)}个基因")
            
            # 将这些基因添加到列表中
            all_u_genes.extend(u_module_genes)
            all_v_genes.extend(v_module_genes)
        
        # 仅保留这些基因
        u_data = u_data[:, all_u_genes].copy()
        v_data = v_data[:, all_v_genes].copy()
    
    else:
        print("警告: 未提供同源基因列表或模块信息，将使用所有基因的交集")
        common_genes = list(set(u_data.var_names) & set(v_data.var_names))
        u_data = u_data[:, common_genes].copy()
        v_data = v_data[:, common_genes].copy()
    
    # 合并数据
    merged_adata = anndata.concat(
        [u_data, v_data],
        join='inner',  # 仅保留两者都有的基因
        label='species',
        keys=['U', 'V']
    )
    
    print(f"合并后数据: {merged_adata.shape[0]}个细胞, {merged_adata.shape[1]}个基因")
    
    return merged_adata 