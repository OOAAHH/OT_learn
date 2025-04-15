"""
多模态CellOT模型 - 支持多样本、多时间点和多基因模块的最优传输模型

核心思想:
1. 扩展基础CellOT模型，通过整合多个样本(u₁...uₙ和v₁...vₘ)的数据进行联合训练
2. 在损失函数中加入样本、时间和模块的约束，引导传输过程
3. 训练一个全局的对偶势函数和传输映射，能够处理不完整的样本并进行补全
"""

import torch
from collections import namedtuple
import numpy as np
from pathlib import Path

from celltosim.networks.icnns import ICNN
from celltosim.models.cellot import FGPair, load_networks, load_opts, load_cellot_model


def compute_multimodal_loss_g(f, g, source, meta_source=None, transport=None, 
                              sample_weight=0.0, time_weight=0.0, module_weight=0.0):
    """
    计算多模态g网络的损失函数，加入样本、时间和模块约束
    
    Args:
        f: f网络
        g: g网络
        source: 源分布样本
        meta_source: 源分布元数据(样本、时间、模块信息)
        transport: 传输后的样本(如果为None则计算)
        sample_weight: 样本约束权重
        time_weight: 时间约束权重
        module_weight: 模块约束权重
    Returns:
        g网络的损失值
    """
    if transport is None:
        transport = g.transport(source)
    
    # 基础损失 - 标准的OT损失
    base_loss = f(transport) - torch.multiply(source, transport).sum(-1, keepdim=True)
    
    # 初始化额外损失
    extra_loss = 0.0
    
    # 如果提供了元数据，加入额外约束
    if meta_source is not None:
        # 样本约束 - 鼓励相同样本保持相似性
        if sample_weight > 0 and 'sample' in meta_source:
            sample_indices = meta_source['sample']
            # 计算样本内部相似性损失
            batch_size = source.size(0)
            for i in range(batch_size):
                # 找到与当前样本相同样本标签的所有样本
                same_sample_mask = (sample_indices == sample_indices[i])
                if same_sample_mask.sum() > 1:  # 至少有另一个相同样本
                    # 计算当前样本与相同样本的传输映射之间的差异
                    same_sample_transport = transport[same_sample_mask]
                    # 计算均值传输
                    mean_transport = same_sample_transport.mean(dim=0, keepdim=True)
                    # 计算当前样本传输与均值传输的距离
                    sample_diff = torch.norm(transport[i] - mean_transport, p=2)
                    extra_loss = extra_loss + sample_weight * sample_diff
        
        # 时间约束 - 鼓励相同时间点的细胞映射到相似位置
        if time_weight > 0 and 'time' in meta_source:
            time_indices = meta_source['time']
            # 类似样本约束，计算时间内部相似性损失
            batch_size = source.size(0)
            for i in range(batch_size):
                same_time_mask = (time_indices == time_indices[i])
                if same_time_mask.sum() > 1:
                    same_time_transport = transport[same_time_mask]
                    mean_transport = same_time_transport.mean(dim=0, keepdim=True)
                    time_diff = torch.norm(transport[i] - mean_transport, p=2)
                    extra_loss = extra_loss + time_weight * time_diff
    
    # 组合损失
    total_loss = base_loss + extra_loss
    
    return total_loss


def compute_multimodal_loss_f(f, g, source, target, meta_source=None, meta_target=None, transport=None,
                             sample_weight=0.0, time_weight=0.0, module_weight=0.0):
    """
    计算多模态f网络的损失函数，加入样本、时间和模块约束
    
    Args:
        f: f网络
        g: g网络
        source: 源分布样本
        target: 目标分布样本
        meta_source: 源分布元数据
        meta_target: 目标分布元数据
        transport: 传输后的样本(如果为None则计算)
        sample_weight: 样本约束权重
        time_weight: 时间约束权重
        module_weight: 模块约束权重
    Returns:
        f网络的损失值
    """
    if transport is None:
        transport = g.transport(source)
    
    # 处理批次大小不匹配的情况
    source_batch_size = source.size(0)
    target_batch_size = target.size(0)
    
    # 如果批次大小不同，随机采样或重复目标数据使其与源数据大小匹配
    if source_batch_size != target_batch_size:
        if target_batch_size > source_batch_size:
            # 随机选择与源数据相同数量的目标数据
            indices = torch.randperm(target_batch_size)[:source_batch_size]
            target = target[indices]
            if meta_target is not None:
                meta_target = {k: v[indices] if torch.is_tensor(v) else v for k, v in meta_target.items()}
        else:
            # 采样或重复目标数据以匹配源数据大小
            indices = torch.randint(0, target_batch_size, (source_batch_size,))
            target = target[indices]
            if meta_target is not None:
                meta_target = {k: v[indices] if torch.is_tensor(v) else v for k, v in meta_target.items()}
    
    # 基础损失 - 标准的OT损失
    base_loss = -f(transport) + f(target)
    
    # 初始化额外损失
    extra_loss = 0.0
    
    # 如果提供了元数据，加入额外约束
    if meta_source is not None and meta_target is not None:
        # 时间对齐约束 - 鼓励相同时间点的源和目标细胞对齐
        if time_weight > 0 and 'time' in meta_source and 'time' in meta_target:
            source_times = meta_source['time']
            target_times = meta_target['time']
            
            # 对每个批次中的样本
            batch_size = source.size(0)
            for i in range(batch_size):
                source_time = source_times[i]
                target_time = target_times[i]
                
                # 如果源和目标时间点不同，增加额外损失
                if source_time != target_time:
                    # 找到与目标时间点相同的源样本
                    same_target_time_mask = (source_times == target_time)
                    
                    if same_target_time_mask.sum() > 0:
                        # 计算当前样本传输与该时间点样本传输的平均差异
                        same_target_time_transport = transport[same_target_time_mask]
                        mean_transport = same_target_time_transport.mean(dim=0, keepdim=True)
                        time_align_diff = torch.norm(transport[i] - mean_transport, p=2)
                        extra_loss = extra_loss + time_weight * time_align_diff
    
    # 组合损失
    total_loss = base_loss + extra_loss
    
    return total_loss


def compute_reconstruction_loss(original, reconstructed, mask=None):
    """
    计算重建损失，用于评估模型的重建/补全能力
    
    Args:
        original: 原始数据
        reconstructed: 重建数据
        mask: 掩码，指示哪些位置需要计算损失
    Returns:
        重建损失
    """
    if mask is not None:
        # 仅计算掩码位置的损失
        mse = ((original - reconstructed) ** 2) * mask
        return mse.sum() / (mask.sum() + 1e-8)
    else:
        # 计算所有位置的损失
        return torch.mean((original - reconstructed) ** 2)


def load_multimodal_cellot_model(config, restore=None, **kwargs):
    """
    加载多模态CellOT模型
    
    Args:
        config: 模型配置
        restore: 模型检查点路径
        kwargs: 额外参数
    Returns:
        (f,g): 网络对
        opts: 优化器对
    """
    # 使用基础CellOT模型的加载函数
    return load_cellot_model(config, restore, **kwargs)


def reconstruct_sample(g, reference_data, target_sample_id, metadata, 
                      sample_key=None, time_key=None, 
                      mask_ratio=0.0, device=None):
    """
    重建/补全缺失的样本
    
    Args:
        g: 训练好的g网络
        reference_data: 参考数据集
        target_sample_id: 目标样本ID
        metadata: 数据集元数据
        sample_key: 样本标签列名
        time_key: 时间标签列名
        mask_ratio: 随机掩码的比例(用于测试补全效果)
        device: 运行设备
    Returns:
        reconstructed: 重建的样本
        original: 原始样本(如果存在)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    g.to(device)
    g.eval()
    
    # 查找目标样本
    if sample_key is None:
        sample_key = metadata.get('sample_key')
    
    if time_key is None:
        time_key = metadata.get('time_key')
    
    # 提取目标样本的数据
    if hasattr(reference_data, 'get_sample_indices'):
        # 使用MultiModalAnnDataset
        sample_indices = reference_data.get_sample_indices(target_sample_id)
        if len(sample_indices) == 0:
            raise ValueError(f"未找到样本ID: {target_sample_id}")
        
        sample_data = reference_data.adata[sample_indices].copy()
    else:
        # 使用普通AnnData
        if sample_key not in reference_data.obs:
            raise ValueError(f"数据集中未找到样本键: {sample_key}")
        
        sample_mask = reference_data.obs[sample_key] == target_sample_id
        if sample_mask.sum() == 0:
            raise ValueError(f"未找到样本ID: {target_sample_id}")
        
        sample_data = reference_data[sample_mask].copy()
    
    # 转换为张量
    original = torch.tensor(sample_data.X, dtype=torch.float32).to(device)
    
    # 应用掩码（如果需要）
    if mask_ratio > 0:
        mask = torch.rand_like(original) > mask_ratio
        masked_data = original.clone()
        masked_data = masked_data * mask
    else:
        masked_data = original
    
    # 重建
    with torch.no_grad():
        masked_data = masked_data.requires_grad_(True)
        with torch.enable_grad():
            reconstructed = g.transport(masked_data)
    
    # 返回重建结果
    return reconstructed.detach().cpu(), original.cpu()


def get_sample_time_matrix(dataset, sample_key, time_key):
    """
    创建样本-时间矩阵，标识每个样本覆盖的时间点
    
    Args:
        dataset: AnnData数据集
        sample_key: 样本标签列名
        time_key: 时间标签列名
    Returns:
        样本-时间矩阵，行为样本，列为时间点
    """
    if not hasattr(dataset, 'obs'):
        if hasattr(dataset, 'adata'):
            dataset = dataset.adata
        else:
            raise ValueError("输入数据集必须是AnnData对象或包含adata属性")
    
    if sample_key not in dataset.obs or time_key not in dataset.obs:
        raise ValueError(f"数据集中未找到样本键({sample_key})或时间键({time_key})")
    
    # 获取唯一的样本和时间点
    samples = sorted(dataset.obs[sample_key].unique())
    times = sorted(dataset.obs[time_key].unique())
    
    # 创建样本-时间矩阵
    sample_time_matrix = np.zeros((len(samples), len(times)), dtype=np.int32)
    
    for i, sample in enumerate(samples):
        for j, time in enumerate(times):
            # 检查该样本是否包含该时间点
            mask = (dataset.obs[sample_key] == sample) & (dataset.obs[time_key] == time)
            sample_time_matrix[i, j] = mask.sum()
    
    return sample_time_matrix, samples, times


def get_missing_samples(dataset_u, dataset_v, sample_key, time_key):
    """
    找出物种U和物种V中的缺失样本和缺失的时间点
    
    Args:
        dataset_u: 物种U的数据集
        dataset_v: 物种V的数据集
        sample_key: 样本标签列名
        time_key: 时间标签列名
    Returns:
        u_missing_samples: U中的缺失样本
        v_missing_samples: V中的缺失样本
    """
    # 获取U和V的样本-时间矩阵
    u_matrix, u_samples, u_times = get_sample_time_matrix(dataset_u, sample_key, time_key)
    v_matrix, v_samples, v_times = get_sample_time_matrix(dataset_v, sample_key, time_key)
    
    # 找出所有时间点
    all_times = sorted(list(set(u_times) | set(v_times)))
    
    # 找出U中哪些样本缺少某些时间点
    u_missing_samples = []
    for i, sample in enumerate(u_samples):
        missing_times = []
        for time in all_times:
            if time not in u_times or u_matrix[i, u_times.index(time)] == 0:
                missing_times.append(time)
        
        if missing_times:
            u_missing_samples.append((sample, missing_times))
    
    # 找出V中哪些样本缺少某些时间点
    v_missing_samples = []
    for i, sample in enumerate(v_samples):
        missing_times = []
        for time in all_times:
            if time not in v_times or v_matrix[i, v_times.index(time)] == 0:
                missing_times.append(time)
        
        if missing_times:
            v_missing_samples.append((sample, missing_times))
    
    return u_missing_samples, v_missing_samples 