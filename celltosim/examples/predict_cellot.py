#!/usr/bin/env python
"""
使用训练好的CellOT模型进行推理示例脚本
"""

import torch
import numpy as np
import argparse
from pathlib import Path

from celltosim.data import load_anndata, AnnDataDataset, cast_dataset_to_loader
from celltosim.networks import ICNN
from celltosim.utils import load_model, transport_cells, batch_transport


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用CellOT模型进行推理')
    parser.add_argument('--data', type=str, required=True, help='AnnData文件路径')
    parser.add_argument('--source', type=str, required=True, help='源标签')
    parser.add_argument('--transport-key', type=str, default='transport', help='transport列名')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出AnnData文件路径')
    parser.add_argument('--hidden-units', type=int, default=64, help='隐藏层单元数')
    parser.add_argument('--n-layers', type=int, default=4, help='隐藏层数量')
    parser.add_argument('--batch-size', type=int, default=128, help='批处理大小')
    parser.add_argument('--dosage', type=float, default=None, help='插值剂量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    return parser.parse_args()


def create_networks(input_dim, hidden_units, n_layers):
    """创建网络实例"""
    hidden_units = [hidden_units] * n_layers
    
    # f网络
    f = ICNN(
        input_dim=input_dim,
        hidden_units=hidden_units,
        activation="LeakyReLU",
        softplus_W_kernels=False
    )
    
    # g网络
    g = ICNN(
        input_dim=input_dim,
        hidden_units=hidden_units,
        activation="LeakyReLU",
        softplus_W_kernels=False
    )
    
    return f, g


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载数据
    print(f"加载数据: {args.data}")
    adata = load_anndata(args.data)
    
    # 提取源数据
    print(f"提取源数据，源标签: {args.source}")
    source_data = adata[adata.obs[args.transport_key] == args.source].copy()
    
    # 获取输入维度
    input_dim = adata.n_vars
    
    # 创建网络结构
    print(f"创建网络，输入维度: {input_dim}, 隐藏层单元: {args.hidden_units}, 层数: {args.n_layers}")
    f, g = create_networks(input_dim, args.hidden_units, args.n_layers)
    
    # 加载训练好的模型参数
    print(f"加载模型: {args.model}")
    checkpoint = torch.load(args.model)
    f.load_state_dict(checkpoint["f_state"])
    g.load_state_dict(checkpoint["g_state"])
    
    # 设置为评估模式
    f.eval()
    g.eval()
    
    # 进行推理
    print(f"执行推理，剂量: {args.dosage if args.dosage is not None else '1.0 (完全映射)'}")
    if len(source_data) > args.batch_size:
        # 对于大数据集，分批处理
        dataset = AnnDataDataset(source_data)
        loader = cast_dataset_to_loader(dataset, batch_size=args.batch_size, shuffle=False)
        
        all_transported = []
        for batch in loader:
            batch.requires_grad_(True)
            with torch.set_grad_enabled(True):
                transported = g.transport(batch)
            
            if args.dosage is not None:
                transported = (1 - args.dosage) * batch + args.dosage * transported
            
            all_transported.append(transported.detach().numpy())
        
        transported_data = np.concatenate(all_transported, axis=0)
        
        # 创建新的AnnData对象
        output_adata = source_data.copy()
        output_adata.X = transported_data
    else:
        # 对于小数据集，一次性处理
        output_adata = transport_cells(g, source_data, dosage=args.dosage)
    
    # 保存结果
    print(f"保存结果: {args.output}")
    output_adata.write(args.output)
    
    print("推理完成")


if __name__ == "__main__":
    main() 