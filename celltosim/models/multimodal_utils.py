"""
为多模态CellOT模型提供的工具函数
"""

import os
import torch
import numpy as np
import anndata as ad
from typing import Dict, List, Tuple, Union, Optional

from celltosim.models.icnn import ICNNSurrogateKR, ICNNGenerator


def save_multimodal_cellot_model(
    f: ICNNSurrogateKR, 
    g: ICNNGenerator, 
    path: str,
    state_dict_only: bool = False
) -> None:
    """
    保存多模态CellOT模型
    
    参数:
        f: 势函数网络
        g: 生成器网络
        path: 保存路径
        state_dict_only: 是否只保存状态字典
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if state_dict_only:
        torch.save({
            'f_state_dict': f.state_dict(),
            'g_state_dict': g.state_dict(),
        }, path)
    else:
        torch.save({
            'f': f,
            'g': g,
            'f_state_dict': f.state_dict(),
            'g_state_dict': g.state_dict(),
        }, path)


def load_multimodal_cellot_model(
    path: str,
    input_dim: Optional[int] = None,
    hidden_dims: int = 64,
    hidden_layers: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[ICNNSurrogateKR, ICNNGenerator]:
    """
    加载多模态CellOT模型
    
    参数:
        path: 模型保存路径
        input_dim: 输入维度，仅在state_dict_only=True时需要
        hidden_dims: 隐藏层维度，仅在state_dict_only=True时需要
        hidden_layers: 隐藏层数量，仅在state_dict_only=True时需要
        device: 设备
        
    返回:
        f: 势函数网络
        g: 生成器网络
    """
    checkpoint = torch.load(path, map_location=device)
    
    if 'f' in checkpoint and 'g' in checkpoint:
        # 完整模型被保存
        f = checkpoint['f'].to(device)
        g = checkpoint['g'].to(device)
    else:
        # 只有状态字典被保存
        if input_dim is None:
            raise ValueError("当加载仅包含状态字典的模型时，必须提供input_dim参数")
        
        f = ICNNSurrogateKR(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            hidden_layers=hidden_layers
        ).to(device)
        g = ICNNGenerator(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            hidden_layers=hidden_layers
        ).to(device)
        
        f.load_state_dict(checkpoint['f_state_dict'])
        g.load_state_dict(checkpoint['g_state_dict'])
    
    f.eval()
    g.eval()
    
    return f, g


def get_missing_samples(
    source_adata: ad.AnnData, 
    target_adata: ad.AnnData
) -> List[str]:
    """
    获取在目标数据中缺失但在源数据中存在的样本列表
    
    参数:
        source_adata: 源AnnData对象
        target_adata: 目标AnnData对象
        
    返回:
        缺失样本列表
    """
    source_samples = set(source_adata.obs['sample'].unique())
    target_samples = set(target_adata.obs['sample'].unique())
    
    return list(source_samples - target_samples)


def get_sample_indices(
    adata: ad.AnnData, 
    sample_name: str
) -> np.ndarray:
    """
    获取特定样本在AnnData对象中的索引
    
    参数:
        adata: AnnData对象
        sample_name: 样本名称
        
    返回:
        样本索引的numpy数组
    """
    return np.where(adata.obs['sample'] == sample_name)[0]


def reconstruct_sample(
    f: ICNNSurrogateKR,
    g: ICNNGenerator,
    source_adata: ad.AnnData,
    target_adata: ad.AnnData,
    missing_sample: str,
    batch_size: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> ad.AnnData:
    """
    重构缺失样本
    
    参数:
        f: 势函数网络
        g: 生成器网络
        source_adata: 源AnnData对象
        target_adata: 目标AnnData对象
        missing_sample: 缺失样本名称
        batch_size: 批量大小
        device: 设备
        
    返回:
        重构的AnnData对象
    """
    # 获取缺失样本的索引
    sample_indices = get_sample_indices(source_adata, missing_sample)
    
    if len(sample_indices) == 0:
        raise ValueError(f"在源数据中未找到样本 {missing_sample}")
    
    # 提取样本数据
    sample_data = source_adata[sample_indices]
    X = sample_data.X
    
    # 确保X是numpy数组
    if isinstance(X, np.ndarray):
        pass
    elif isinstance(X, torch.Tensor):
        X = X.numpy()
    else:
        try:
            X = X.toarray()  # 对于稀疏矩阵
        except:
            pass
    
    # 转换为PyTorch张量
    X = torch.from_numpy(X).float()
    
    # 分批处理
    reconstructed_data = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size].to(device)
            recon_batch = g(f(batch)).cpu()
            reconstructed_data.append(recon_batch)
    
    # 合并所有批次
    reconstructed_tensor = torch.cat(reconstructed_data, dim=0)
    
    # 转换回numpy
    reconstructed_array = reconstructed_tensor.numpy()
    
    # 创建一个新的AnnData对象，包含重构的数据
    reconstructed_adata = ad.AnnData(
        X=reconstructed_array,
        obs=sample_data.obs.copy(),
        var=target_adata.var.copy() if target_adata.shape[1] == reconstructed_array.shape[1] else None
    )
    
    return reconstructed_adata


def get_modality_indices(
    adata: ad.AnnData, 
    modality: str
) -> np.ndarray:
    """
    获取特定模态在AnnData对象中的特征索引
    
    参数:
        adata: AnnData对象
        modality: 模态名称
        
    返回:
        模态索引的numpy数组
    """
    if 'modality' not in adata.var:
        raise ValueError("AnnData对象的var中没有'modality'列")
    
    return np.where(adata.var['modality'] == modality)[0]


def extract_modality_data(
    adata: ad.AnnData, 
    modality: str
) -> ad.AnnData:
    """
    从AnnData对象中提取特定模态的数据
    
    参数:
        adata: AnnData对象
        modality: 模态名称
        
    返回:
        包含特定模态数据的AnnData对象
    """
    modality_indices = get_modality_indices(adata, modality)
    return adata[:, modality_indices]


def combine_modalities(
    adatas: List[ad.AnnData], 
    modality_names: List[str]
) -> ad.AnnData:
    """
    将多个模态的AnnData对象合并为一个多模态AnnData对象
    
    参数:
        adatas: AnnData对象列表
        modality_names: 对应模态名称列表
        
    返回:
        合并后的多模态AnnData对象
    """
    if len(adatas) != len(modality_names):
        raise ValueError("AnnData对象列表和模态名称列表长度必须一致")
    
    # 确保所有AnnData对象具有相同的细胞数量和顺序
    n_obs = adatas[0].n_obs
    for adata in adatas[1:]:
        if adata.n_obs != n_obs:
            raise ValueError("所有AnnData对象必须具有相同的细胞数量")
    
    # 合并所有AnnData对象
    combined = ad.concat(adatas, axis=1, join='outer', label='modality', keys=modality_names)
    
    return combined 