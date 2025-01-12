"""
整体思路
这个脚本基于最优传输（Optimal Transport）中的对偶理论，使用深度学习框架（以 ICNN 为核心）来求解从源分布 (source) 到目标分布 (target) 的最优映射。
关键思想是引入两个网络 f 和 g：
• f 网络对应对偶问题中的一个势函数 (Kantorovich potential)，并且要求 f 为凸函数；
• g 网络用于显式给出从源分布到目标分布的传输映射，简而言之就是“怎么把源分布点 x 映射到目标分布 y”。
为了保证凸性，这里采用了 ICNN（Input Convex Neural Network）结构来构造 f 和 g。
ICNN 网络天然可以确保输出函数对输入是凸的或半凸的。
通过对这两个网络的联合训练，一方面可以计算对偶问题中的势函数 f，
另一方面可以得到从源分布到目标分布的映射 g。
最后，通过这两个网络的输出，可以估计和逼近 Wasserstein 距离（或其它形式的最优传输距离）。
"""


from pathlib import Path
import torch
from collections import namedtuple
from cellot.networks.icnns import ICNN

from absl import flags

FLAGS = flags.FLAGS

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
    • 读取用户/配置文件所提供的网络结构参数。
    • 从配置中分别创建两个 ICNN 网络 f 和 g。
    • f 用于代表对偶势函数，g 用于实际的传输映射。
    • 通过 unpack_kernel_init_fxn 函数，可以根据配置选择不同的参数初始化方式。
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
        **fkwargs.pop("kernel_init_fxn")
    )

    # 配置g网络参数
    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
        **gkwargs.pop("kernel_init_fxn")
    )

    f = ICNN(**fkwargs)
    g = ICNN(**gkwargs)

    if "verbose" in FLAGS and FLAGS.verbose:
        print(g)
        print(kwargs)

    return f, g


def load_opts(config, f, g):
    """
    为f和g网络加载优化器
    Args:
        config: 优化器配置
        f, g: ICNN网络实例
    Returns:
        opts: 包含f和g优化器的FGPair实例
    • 为 f 和 g 两个网络分别初始化优化器 (Adam)。
    • 可以在配置中分别指定 f 和 g 的优化器超参数，比如学习率 (lr)、beta1、beta2 等。
    """
    kwargs = dict(config.get("optim", {}))
    assert kwargs.pop("optimizer", "Adam") == "Adam"

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
    • 这是一个高层接口，用来一键加载网络和优化器。
    • 如果提供了 restore 路径，会加载之前训练的检查点 (checkpoint)，从而继续某个中断的训练。
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
        f, g: ICNN网络
        source: 源分布样本
        transport: 传输后的样本
    Returns:
        g网络的损失值
    • 在最优传输的对偶形式中，g 需要使得 f 函数和传输映射相匹配。
    • 可以把这看作是约束“传输映射得到的结果要和对偶势函数保持一致”的一部分。
    """
    if transport is None:
        transport = g.transport(source)

    return f(transport) - torch.multiply(source, transport).sum(-1, keepdim=True)


def compute_g_constraint(g, form=None, beta=0):
    """
    计算g网络的约束项
    Args:
        g: g网络
        form: 约束形式
        beta: 约束强度
    Returns:
        约束损失值
    • 对 g 进行一些额外的约束，比如权重的范数约束、或把权重截断到一定范围内，这在 ICNN 中是常见的技巧，用于确保网络凸性或防止权重发散。
    • 当 form="clamp" 时，会调用 g.clamp_w() 将权重剪裁；当 form="fnorm" 时，则添加基于 Frobenius 范数的正则项。
    """
    if form is None or form == "None":
        return 0

    if form == "clamp":
        g.clamp_w()
        return 0

    elif form == "fnorm":
        if beta == 0:
            return 0
        return beta * sum(map(lambda w: w.weight.norm(p="fro"), g.W))

    raise ValueError


def compute_loss_f(f, g, source, target, transport=None):
    """
    计算f网络的损失函数
    Args:
        f, g: ICNN网络
        source: 源分布样本
        target: 目标分布样本
        transport: 传输后的样本
    Returns:
        f网络的损失值
    • 计算对 f 网络的损失。
    • 对偶问题中，f 需要对目标分布 (target) 和 Source→Transport(g(source)) 进行匹配。
    • 这个损失也能看作是保证 f 正确反映源和目标之间的潜在运输代价。
    """
    if transport is None:
        transport = g.transport(source)

    return -f(transport) + f(target)


def compute_w2_distance(f, g, source, target, transport=None):
    """
    计算Wasserstein-2距离
    Args:
        f, g: ICNN网络
        source: 源分布样本
        target: 目标分布样本
        transport: 传输后的样本
    Returns:
        W2距离估计值
    • 使用前面训练好的 f 和 g，去估计 Wasserstein-2 距离（W2 distance）。
    • 在理论上，这个值就是最优传输代价。
    • 这里用 f(network_output) - x·g(x) + 一些代价项等来得到对 W2 的估计。
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


def numerical_gradient(param, fxn, *args, eps=1e-4):
    """
    数值计算梯度
    Args:
        param: 需要计算梯度的参数
        fxn: 目标函数
        args: 函数参数
        eps: 扰动大小
    Returns:
        数值梯度
    • 一个简单的数值求导小工具，用来对某些标量函数 fxn 做数值梯度检查或者调试。
    • 通过对 param 进行正负方向上的微小扰动 (±eps)，计算函数值差分来近似梯度。
    """
    with torch.no_grad():
        param += eps
    plus = float(fxn(*args))

    with torch.no_grad():
        param -= 2 * eps
    minus = float(fxn(*args))

    with torch.no_grad():
        param += eps

    return (plus - minus) / (2 * eps)

"""
关键技术点总结
使用 ICNN (Input Convex Neural Network)：
• 在最优传输的对偶形式中，需要势函数 f(x) 对 x 可微并且是凸函数。ICNN 是一类特殊设计的网络，使得网络输出随输入凸或半凸。
• 通过隐式或显式的约束（比如对权重加非负约束或其他特殊结构），保证网络的凸性。
Kantorovich 对偶：
• 最优传输的 Kantorovich 对偶告诉我们，W2 距离可以用对偶变量 f 来表示，满足一定的不等式约束。
• 具体在代码里，f 用来估计对偶势，g 是运输映射，相当于用深度网络实现“往目标分布搬运”的过程。
训练方式：
• 分别定义了 compute_loss_f() 和 compute_loss_g()，实现了对偶的损失配套。
• 优化时会同时更新 f 和 g，使得它们在对偶条件下互相满足，从而收敛到最优传输。
"""

"""
总体流程
读取和解析配置 (config)。
创建 f 和 g 两个 ICNN (load_networks)。
创建优化器 (load_opts)。
如果提供了之前训练的 checkpoint，则加载网络和优化器的状态 (load_cellot_model)。
在训练循环中，根据 compute_loss_f() 和 compute_loss_g() 计算损失并反向传播，更新 f 和 g。
训练完后，可以用 compute_w2_distance() 来评估训练效果，或用 g.transport() 来查看传输映射后的结果。
"""

"""
从高层来讲，它依赖“在对偶理论下，对分布映射进行深度学习”的想法，并通过 ICNN 的凸性为对偶势函数提供模型保证，从而完成对最优传输的学习过程。
"""