"""
多模态CellOT模型训练函数
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

from celltosim.models import (
    compute_multimodal_loss_f, 
    compute_multimodal_loss_g, 
    compute_reconstruction_loss,
    compute_w2_distance
)


def check_loss(*args):
    """
    检查损失值是否为NaN
    
    Args:
        *args: 要检查的损失值
    Raises:
        ValueError: 如果发现NaN值
    """
    for arg in args:
        if torch.isnan(arg):
            raise ValueError("损失值为NaN")


def train_multimodal_cellot(
    f, g, loaders, outdir, 
    n_iters=5000, 
    n_inner_iters=5,
    eval_freq=50, 
    logs_freq=10, 
    cache_freq=100,
    sample_weight=0.1,
    time_weight=0.1,
    module_weight=0.0,
    restore=None
):
    """
    训练多模态CellOT模型
    
    Args:
        f: f网络
        g: g网络
        loaders: 数据加载器命名元组，包含train_source等
        outdir: 输出目录
        n_iters: 训练迭代次数
        n_inner_iters: 内部迭代次数
        eval_freq: 评估频率
        logs_freq: 日志记录频率
        cache_freq: 缓存频率
        sample_weight: 样本约束权重
        time_weight: 时间约束权重
        module_weight: 模块约束权重
        restore: 恢复训练的检查点路径
    Returns:
        训练结果
    """
    def state_dict(f, g, opts):
        """创建包含模型状态的字典"""
        return {
            "f_state": f.state_dict(),
            "g_state": g.state_dict(),
            "opt_f_state": opts.f.state_dict(),
            "opt_g_state": opts.g.state_dict(),
        }

    # 创建输出目录
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cachedir = outdir / "cache"
    cachedir.mkdir(exist_ok=True)
    
    # 设置TensorBoard日志记录器
    log_dir = outdir / "logs"
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 创建优化器
    opts = namedtuple('Optimizers', ['f', 'g'])(
        f=torch.optim.Adam(f.parameters(), lr=1e-4),
        g=torch.optim.Adam(g.parameters(), lr=1e-4)
    )
    
    # 如果有检查点，则恢复模型状态
    step = 0
    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore)
        f.load_state_dict(ckpt["f_state"])
        g.load_state_dict(ckpt["g_state"])
        opts.f.load_state_dict(ckpt["opt_f_state"])
        opts.g.load_state_dict(ckpt["opt_g_state"])
        if "step" in ckpt:
            step = ckpt["step"]
    
    # 获取迭代器
    train_source_iter = iter(loaders.train_source)
    train_target_iter = iter(loaders.train_target)
    
    # 创建进度条
    ticker = trange(step, n_iters, initial=step, total=n_iters)
    
    # 记录最小W2距离
    min_w2 = np.inf
    
    # 开始训练循环
    for step in ticker:
        # 训练生成器g（内部循环）
        for _ in range(n_inner_iters):
            try:
                # 获取源数据和元数据
                batch = next(train_source_iter)
                if isinstance(batch, tuple) and len(batch) == 2:
                    source, meta_source = batch
                else:
                    source = batch
                    meta_source = None
            except StopIteration:
                train_source_iter = iter(loaders.train_source)
                batch = next(train_source_iter)
                if isinstance(batch, tuple) and len(batch) == 2:
                    source, meta_source = batch
                else:
                    source = batch
                    meta_source = None
                
            # 设置梯度
            source = source.requires_grad_(True)
            
            # 训练生成器
            opts.g.zero_grad()
            gl = compute_multimodal_loss_g(
                f, g, source, meta_source, 
                sample_weight=sample_weight,
                time_weight=time_weight,
                module_weight=module_weight
            ).mean()
            
            # 添加权重惩罚（如果g网络有这个属性）
            if hasattr(g, 'fnorm_penalty') and not g.softplus_W_kernels and g.fnorm_penalty > 0:
                gl = gl + g.penalize_w()
                
            gl.backward()
            opts.g.step()
        
        # 获取新的源数据和目标数据
        try:
            batch = next(train_source_iter)
            if isinstance(batch, tuple) and len(batch) == 2:
                source, meta_source = batch
            else:
                source = batch
                meta_source = None
        except StopIteration:
            train_source_iter = iter(loaders.train_source)
            batch = next(train_source_iter)
            if isinstance(batch, tuple) and len(batch) == 2:
                source, meta_source = batch
            else:
                source = batch
                meta_source = None
            
        try:
            batch = next(train_target_iter)
            if isinstance(batch, tuple) and len(batch) == 2:
                target, meta_target = batch
            else:
                target = batch
                meta_target = None
        except StopIteration:
            train_target_iter = iter(loaders.train_target)
            batch = next(train_target_iter)
            if isinstance(batch, tuple) and len(batch) == 2:
                target, meta_target = batch
            else:
                target = batch
                meta_target = None
        
        # 设置梯度
        source = source.requires_grad_(True)
        
        # 处理批次大小不匹配的情况（现在由compute_multimodal_loss_f内部处理）
        
        # 训练判别器f
        opts.f.zero_grad()
        fl = compute_multimodal_loss_f(
            f, g, source, target, meta_source, meta_target,
            sample_weight=sample_weight,
            time_weight=time_weight,
            module_weight=module_weight
        ).mean()
        fl.backward()
        opts.f.step()
        
        # 检查损失值
        check_loss(gl, fl)
        
        # 截断权重
        if hasattr(f, 'clamp_w'):
            f.clamp_w()
        
        # 定期记录训练日志
        if step % logs_freq == 0:
            writer.add_scalar('train/g_loss', gl.item(), step)
            writer.add_scalar('train/f_loss', fl.item(), step)
            
            # 更新进度条显示
            ticker.set_description(f"g_loss: {gl.item():.4f}, f_loss: {fl.item():.4f}")
        
        # 定期评估模型
        if step % eval_freq == 0:
            # 获取测试数据
            try:
                batch = next(iter(loaders.test_source))
                if isinstance(batch, tuple) and len(batch) == 2:
                    test_source_batch, test_meta_source = batch
                else:
                    test_source_batch = batch
                    test_meta_source = None
                    
                batch = next(iter(loaders.test_target))
                if isinstance(batch, tuple) and len(batch) == 2:
                    test_target_batch, test_meta_target = batch
                else:
                    test_target_batch = batch
                    test_meta_target = None
            except Exception as e:
                print(f"评估数据加载错误: {e}")
                continue
            
            # 处理测试数据批次大小不匹配的情况
            test_source_size = test_source_batch.size(0)
            test_target_size = test_target_batch.size(0)
            
            if test_source_size != test_target_size:
                # 如果目标批次大小大于源批次大小，随机选择一部分
                if test_target_size > test_source_size:
                    indices = torch.randperm(test_target_size)[:test_source_size]
                    test_target_batch = test_target_batch[indices]
                    if test_meta_target is not None:
                        test_meta_target = {k: v[indices] if torch.is_tensor(v) else v for k, v in test_meta_target.items()}
                # 如果源批次大小大于目标批次大小，随机采样以匹配大小
                else:
                    indices = torch.randint(0, test_target_size, (test_source_size,))
                    test_target_batch = test_target_batch[indices]
                    if test_meta_target is not None:
                        test_meta_target = {k: v[indices] if torch.is_tensor(v) else v for k, v in test_meta_target.items()}
            
            # 计算传输结果 - 注意这部分需要使用梯度，即使在评估模式下
            test_source_batch = test_source_batch.requires_grad_(True)
            
            # 对于transport操作，我们需要计算梯度
            with torch.enable_grad():
                transport = g.transport(test_source_batch)
            
            # 其余评估在no_grad模式下进行
            with torch.no_grad():
                # 计算W2距离
                w2 = compute_w2_distance(f, g, test_source_batch, test_target_batch, transport).item()
                writer.add_scalar('eval/w2_distance', w2, step)
                
                # 计算重建损失
                recon_loss = compute_reconstruction_loss(test_source_batch, transport).item()
                writer.add_scalar('eval/recon_loss', recon_loss, step)
                
                # 如果是最小W2距离，则保存模型
                if w2 < min_w2:
                    min_w2 = w2
                    torch.save(
                        {**state_dict(f, g, opts), "step": step, "min_w2": min_w2},
                        cachedir / "best_model.pt"
                    )
        
        # 定期保存检查点
        if step % cache_freq == 0:
            torch.save(
                {**state_dict(f, g, opts), "step": step},
                cachedir / "last.pt"
            )
    
    # 保存最终模型
    torch.save(
        {**state_dict(f, g, opts), "step": step},
        cachedir / "final_model.pt"
    )
    
    writer.close()
    
    return f, g 