"""
CellOT - 基于最优传输的单细胞预测模型

核心思想:
1. 使用Input Convex Neural Network (ICNN)构建最优传输模型
2. 两个关键网络: f(势函数)和g(传输映射)
3. 通过f和g的联合训练实现从源分布到目标分布的最优映射
"""

import torch
from collections import namedtuple
from pathlib import Path
from celltosim.networks.icnns import ICNN

# 定义一个命名元组用于存储f和g网络的配对
FGPair = namedtuple("FGPair", "f g")


def load_networks(config, **kwargs):
    """
    根据配置加载f和g两个ICNN网络
    
    Args:
        config: 包含网络配置的字典
        kwargs: 额外的网络参数
    Returns:
        f, g: 两个ICNN网络实例
    """
    def unpack_kernel_init_fxn(name="uniform", **kwargs):
        """解析核初始化函数"""
        if name == "normal":
            def init(*args):
                return torch.nn.init.normal_(*args, **kwargs)
        elif name == "uniform":
            def init(*args):
                return torch.nn.init.uniform_(*args, **kwargs)
        else:
            raise ValueError
        return init

    kwargs.setdefault("hidden_units", [64] * 4)
    kwargs.update(dict(config.get("model", {})))

    # 处理f和g网络的特定参数
    kwargs.pop("name")
    if "latent_dim" in kwargs:
        kwargs.pop("latent_dim")
    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})

    # 配置f网络参数
    fkwargs = kwargs.copy()
    fkwargs.update(fupd)
    fkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
        **fkwargs.pop("kernel_init_fxn", {"name": "uniform"})
    )

    # 配置g网络参数
    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
        **gkwargs.pop("kernel_init_fxn", {"name": "uniform"})
    )

    f = ICNN(**fkwargs)
    g = ICNN(**gkwargs)

    return f, g


def load_opts(config, f, g):
    """
    为f和g网络加载优化器
    
    Args:
        config: 优化器配置
        f, g: ICNN网络实例
    Returns:
        opts: 包含f和g优化器的FGPair实例
    """
    kwargs = dict(config.get("optim", {}))
    optimizer_name = kwargs.pop("optimizer", "Adam")
    assert optimizer_name == "Adam", f"仅支持Adam优化器，当前配置为: {optimizer_name}"

    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})

    # 配置f网络优化器参数
    fkwargs = kwargs.copy()
    fkwargs.update(fupd)
    fkwargs["betas"] = (fkwargs.pop("beta1", 0.9), fkwargs.pop("beta2", 0.999))

    # 配置g网络优化器参数
    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["betas"] = (gkwargs.pop("beta1", 0.9), gkwargs.pop("beta2", 0.999))

    opts = FGPair(
        f=torch.optim.Adam(f.parameters(), **fkwargs),
        g=torch.optim.Adam(g.parameters(), **gkwargs),
    )

    return opts


def load_cellot_model(config, restore=None, **kwargs):
    """
    加载完整的CellOT模型,包括网络和优化器
    
    Args:
        config: 模型配置
        restore: 模型检查点路径
        kwargs: 额外参数
    Returns:
        (f,g): 网络对
        opts: 优化器对
    """
    f, g = load_networks(config, **kwargs)
    opts = load_opts(config, f, g)

    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore)
        f.load_state_dict(ckpt["f_state"])
        opts.f.load_state_dict(ckpt["opt_f_state"])

        g.load_state_dict(ckpt["g_state"])
        opts.g.load_state_dict(ckpt["opt_g_state"])

    return (f, g), opts


def compute_loss_g(f, g, source, transport=None):
    """
    计算g网络的损失函数
    
    Args:
        f: f网络
        g: g网络
        source: 源分布样本
        transport: 传输后的样本(如果为None则计算)
    Returns:
        g网络的损失值
    """
    if transport is None:
        transport = g.transport(source)
    
    # 基础损失
    base_loss = f(transport) - torch.multiply(source, transport).sum(-1, keepdim=True)
    
    return base_loss


def compute_loss_f(f, g, source, target, transport=None):
    """
    计算f网络的损失函数
    
    Args:
        f: f网络
        g: g网络
        source: 源分布样本
        target: 目标分布样本
        transport: 传输后的样本(如果为None则计算)
    Returns:
        f网络的损失值
    """
    if transport is None:
        transport = g.transport(source)
    
    # 基础损失
    base_loss = -f(transport) + f(target)
    
    return base_loss


def compute_w2_distance(f, g, source, target, transport=None):
    """
    计算Wasserstein-2距离
    
    Args:
        f: f网络
        g: g网络
        source: 源分布样本
        target: 目标分布样本
        transport: 传输后的样本(如果为None则计算)
    Returns:
        W2距离估计值
    """
    if transport is None:
        transport = g.transport(source).squeeze()

    with torch.no_grad():
        Cpq = (source * source).sum(1, keepdim=True) + (target * target).sum(
            1, keepdim=True
        )
        Cpq = 0.5 * Cpq

        cost = (
            f(transport)
            - torch.multiply(source, transport).sum(-1, keepdim=True)
            - f(target)
            + Cpq
        )
        cost = cost.mean()
    return cost 