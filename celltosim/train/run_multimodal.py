"""
运行多模态CellOT训练的入口脚本
"""

import os
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from collections import namedtuple

from celltosim.data import (
    MultiModalAnnDataset,
    prepare_multimodal_data,
    merge_species_data
)
from celltosim.models import (
    ICNNSurrogateKR,
    ICNNGenerator,
    load_multimodal_cellot_model,
    reconstruct_sample,
    get_missing_samples
)
from celltosim.train.multimodal_train import train_multimodal_cellot


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练多模态CellOT模型")
    
    # 数据相关参数
    parser.add_argument("--source", type=str, required=True, help="源AnnData文件路径")
    parser.add_argument("--target", type=str, required=True, help="目标AnnData文件路径")
    parser.add_argument("--batch_size", type=int, default=128, help="批量大小")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载器工作线程数")
    
    # 模型相关参数
    parser.add_argument("--hidden_dims", type=int, default=64, help="隐藏维度")
    parser.add_argument("--hidden_layers", type=int, default=3, help="隐藏层数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    
    # 训练相关参数
    parser.add_argument("--outdir", type=str, default="results", help="输出目录")
    parser.add_argument("--n_iters", type=int, default=5000, help="训练迭代次数")
    parser.add_argument("--n_inner_iters", type=int, default=5, help="内部迭代次数")
    parser.add_argument("--eval_freq", type=int, default=50, help="评估频率")
    parser.add_argument("--logs_freq", type=int, default=10, help="日志记录频率")
    parser.add_argument("--cache_freq", type=int, default=100, help="缓存频率")
    parser.add_argument("--sample_weight", type=float, default=0.1, help="样本约束权重")
    parser.add_argument("--time_weight", type=float, default=0.1, help="时间约束权重")
    parser.add_argument("--module_weight", type=float, default=0.0, help="模块约束权重")
    parser.add_argument("--restore", type=str, default=None, help="恢复训练的检查点路径")
    
    # 多模态特定参数
    parser.add_argument("--merge_species", action="store_true", help="是否合并不同物种数据")
    parser.add_argument("--homologous_genes", type=str, default=None, help="同源基因文件路径，如果需要合并物种")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / timestamp
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 准备数据
    print("准备数据...")
    if args.merge_species:
        if args.homologous_genes is None:
            raise ValueError("合并物种数据需要提供同源基因文件路径")
        
        merged_data = merge_species_data(
            args.source,
            args.target,
            homologous_genes_file=args.homologous_genes
        )
        source_adata, target_adata = merged_data['source'], merged_data['target']
    else:
        # 直接加载数据
        source_adata, target_adata = prepare_multimodal_data(args.source, args.target)
    
    # 创建数据集和数据加载器
    print("创建数据集和数据加载器...")
    source_dataset = MultiModalAnnDataset(source_adata)
    target_dataset = MultiModalAnnDataset(target_adata)
    
    # 划分训练集和测试集
    train_source_dataset, test_source_dataset = torch.utils.data.random_split(
        source_dataset, [0.8, 0.2]
    )
    train_target_dataset, test_target_dataset = torch.utils.data.random_split(
        target_dataset, [0.8, 0.2]
    )
    
    # 创建数据加载器
    train_source_loader = DataLoader(
        train_source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
    )
    train_target_loader = DataLoader(
        train_target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
    )
    test_source_loader = DataLoader(
        test_source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
    )
    test_target_loader = DataLoader(
        test_target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
    )
    
    # 创建命名元组存储数据加载器
    loaders = namedtuple(
        "Loaders", ["train_source", "train_target", "test_source", "test_target"]
    )(
        train_source=train_source_loader,
        train_target=train_target_loader,
        test_source=test_source_loader,
        test_target=test_target_loader,
    )
    
    # 确定输入维度
    input_dim = source_dataset[0][0].shape[0] if isinstance(source_dataset[0], tuple) else source_dataset[0].shape[0]
    
    # 创建模型
    print(f"创建模型，输入维度: {input_dim}...")
    f = ICNNSurrogateKR(
        input_dim=input_dim,
        hidden_layers=args.hidden_layers,
        hidden_dims=args.hidden_dims,
    ).to(args.device)
    
    g = ICNNGenerator(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        hidden_layers=args.hidden_layers,
    ).to(args.device)
    
    # 训练模型
    print("开始训练...")
    f, g = train_multimodal_cellot(
        f, g, loaders, outdir,
        n_iters=args.n_iters,
        n_inner_iters=args.n_inner_iters,
        eval_freq=args.eval_freq,
        logs_freq=args.logs_freq,
        cache_freq=args.cache_freq,
        sample_weight=args.sample_weight,
        time_weight=args.time_weight,
        module_weight=args.module_weight,
        restore=args.restore
    )
    
    # 保存模型路径
    model_path = outdir / "cache" / "final_model.pt"
    
    # 查找缺失样本
    print("分析缺失样本...")
    missing_samples = get_missing_samples(source_adata, target_adata)
    
    if missing_samples:
        print(f"发现 {len(missing_samples)} 个缺失样本。重构缺失样本...")
        
        # 重建缺失样本
        for missing in missing_samples:
            print(f"重构样本 {missing}...")
            
            # 加载模型
            loaded_f, loaded_g = load_multimodal_cellot_model(
                model_path, 
                input_dim=input_dim,
                hidden_dims=args.hidden_dims,
                hidden_layers=args.hidden_layers, 
                device=args.device
            )
            
            # 重构样本
            reconstructed = reconstruct_sample(
                loaded_f, 
                loaded_g,
                source_adata, 
                target_adata,
                missing_sample=missing,
                device=args.device
            )
            
            # 保存重构结果
            recon_path = outdir / f"reconstructed_{missing}.h5ad"
            reconstructed.write_h5ad(recon_path)
            print(f"已保存重构结果到 {recon_path}")
    else:
        print("未发现缺失样本。")
    
    print(f"训练完成，结果保存在 {outdir}")


if __name__ == "__main__":
    main() 