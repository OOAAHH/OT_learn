#!/usr/bin/env python
"""
CellOT模型训练示例脚本
"""

import torch
import numpy as np
import argparse
from pathlib import Path

from celltosim.data import load_anndata, prepare_cellot_data
from celltosim.networks import ICNN
from celltosim.train import train_cellot


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练CellOT模型')
    parser.add_argument('--data', type=str, required=True, help='AnnData文件路径')
    parser.add_argument('--source', type=str, required=True, help='源标签')
    parser.add_argument('--target', type=str, required=True, help='目标标签')
    parser.add_argument('--transport-key', type=str, default='transport', help='transport列名')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    parser.add_argument('--hidden-units', type=int, default=64, help='隐藏层单元数')
    parser.add_argument('--n-layers', type=int, default=4, help='隐藏层数量')
    parser.add_argument('--batch-size', type=int, default=128, help='批处理大小')
    parser.add_argument('--n-iters', type=int, default=5000, help='训练迭代次数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--restore', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"加载数据: {args.data}")
    adata = load_anndata(args.data)
    
    # 准备数据集和加载器
    print(f"准备数据集，源标签: {args.source}，目标标签: {args.target}")
    datasets, loaders, input_dim = prepare_cellot_data(
        adata, 
        source_label=args.source, 
        target_label=args.target,
        transport_key=args.transport_key,
        batch_size=args.batch_size
    )
    
    # 创建网络
    print(f"创建网络，输入维度: {input_dim}, 隐藏层单元: {args.hidden_units}, 层数: {args.n_layers}")
    hidden_units = [args.hidden_units] * args.n_layers
    
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
    
    # 训练模型
    print(f"开始训练，迭代次数: {args.n_iters}")
    train_cellot(
        f=f, 
        g=g, 
        loaders=loaders, 
        outdir=output_dir,
        n_iters=args.n_iters,
        restore=args.restore
    )
    
    print(f"训练完成，模型保存在: {output_dir}")


if __name__ == "__main__":
    main() 