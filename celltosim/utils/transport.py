"""
使用训练好的CellOT模型进行传输映射
"""

import torch
import anndata
import numpy as np
from pathlib import Path


def load_model(model_path, f_class, g_class):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        f_class: f网络类
        g_class: g网络类
    Returns:
        f, g: 加载的网络
    """
    model_path = Path(model_path)
    assert model_path.exists(), f"模型文件不存在: {model_path}"
    
    # 加载模型状态
    ckpt = torch.load(model_path)
    
    # 创建模型实例
    f = f_class()
    g = g_class()
    
    # 加载状态
    f.load_state_dict(ckpt["f_state"])
    g.load_state_dict(ckpt["g_state"])
    
    # 设置为评估模式
    f.eval()
    g.eval()
    
    return f, g


def transport_cells(g, source_data, dosage=None):
    """
    对源数据执行传输映射
    
    Args:
        g: g网络
        source_data: 源数据张量或AnnData对象
        dosage: 插值剂量，None表示完全映射
    Returns:
        映射后的数据
    """
    g.eval()
    
    # 处理AnnData对象
    if isinstance(source_data, anndata.AnnData):
        data = torch.tensor(source_data.X, dtype=torch.float32)
        is_anndata = True
    else:
        data = source_data
        is_anndata = False
    
    # 确保数据是浮点型
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    elif data.dtype != torch.float32:
        data = data.float()
    
    # 开启梯度计算
    data.requires_grad_(True)
    
    # 执行传输
    with torch.set_grad_enabled(True):
        transported = g.transport(data)
    
    # 应用插值剂量
    if dosage is not None:
        transported = (1 - dosage) * data + dosage * transported
    
    # 返回结果
    if is_anndata:
        return anndata.AnnData(
            transported.detach().numpy(),
            obs=source_data.obs.copy(),
            var=source_data.var.copy()
        )
    else:
        return transported.detach()


def batch_transport(g, source_loader, batch_size=128, dosage=None):
    """
    分批处理大型数据集
    
    Args:
        g: g网络
        source_loader: 源数据的DataLoader
        batch_size: 批处理大小
        dosage: 插值剂量
    Returns:
        映射后的所有批次数据
    """
    g.eval()
    all_transported = []
    
    for batch in source_loader:
        batch.requires_grad_(True)
        with torch.set_grad_enabled(True):
            transported = g.transport(batch)
        
        if dosage is not None:
            transported = (1 - dosage) * batch + dosage * transported
        
        all_transported.append(transported.detach())
    
    return torch.cat(all_transported, dim=0) 