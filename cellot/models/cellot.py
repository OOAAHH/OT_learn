"""
整体思路
这个脚本基于最优传输（Optimal Transport）中的对偶理论，使用深度学习框架（以 ICNN 为核心）来求解从源分布 (source) 到目标分布 (target) 的最优映射。
关键思想是引入两个网络 f 和 g：
• f 网络对应对偶问题中的一个势函数 (Kantorovich potential)，并且要求 f 为凸函数；
• g 网络用于显式给出从源分布到目标分布的传输映射，简而言之就是"怎么把源分布点 x 映射到目标分布 y"。
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


def compute_loss_g(f, g, source, transport=None, 
                  module_weights=None, time_weights=None):
    """扩展g网络的损失函数，加入module和time的权重"""
    if transport is None:
        transport = g.transport(source)
    
    # 基础损失
    base_loss = f(transport) - torch.multiply(source, transport).sum(-1, keepdim=True)
    
    # 添加module特异的损失
    if module_weights is not None:
        module_loss = compute_module_specific_loss(g, source, transport, module_weights)
        base_loss = base_loss + module_loss
        
    # 添加时间特异的损失
    if time_weights is not None:
        time_loss = compute_time_specific_loss(g, source, transport, time_weights)
        base_loss = base_loss + time_loss
        
    return base_loss


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


def compute_loss_f(f, g, source, target, transport=None, 
                  module_weights=None, time_weights=None):
    """扩展f网络的损失函数，加入module和time的权重"""
    if transport is None:
        transport = g.transport(source)
    
    # 基础损失
    base_loss = -f(transport) + f(target)
    
    # 如果提供了module权重，加入module特异的损失项
    if module_weights is not None:
        module_loss = compute_module_specific_loss(f, transport, target, module_weights)
        base_loss = base_loss + module_loss
        
    # 如果提供了时间权重，加入时间特异的损失项
    if time_weights is not None:
        time_loss = compute_time_specific_loss(f, transport, target, time_weights)
        base_loss = base_loss + time_loss
        
    return base_loss


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


def compute_module_specific_loss(f_or_g, source, target, module_weights):
    """计算模块特异的损失"""
    loss = 0
    for module, weight in module_weights.items():
        # 获取该module的基因索引
        module_idx = get_module_indices(module)
        # 计算该module的特异损失
        module_loss = compute_submatrix_loss(f_or_g, source[:, module_idx], 
                                           target[:, module_idx])
        loss += weight * module_loss
    return loss


def compute_time_specific_loss(f_or_g, source, target, time_weights):
    """计算时间特异的损失"""
    loss = 0
    for time, weight in time_weights.items():
        # 获取该时间点的细胞索引
        time_idx = get_time_indices(time)
        # 计算该时间点的特异损失
        time_loss = compute_submatrix_loss(f_or_g, source[time_idx], 
                                         target[time_idx])
        loss += weight * time_loss
    return loss


def compute_cell_type_consistency(source, target, transport, cell_labels):
    """计算细胞类型的一致性指标（不是损失函数的一部分，而是评估指标）"""
    consistency_scores = {}
    for cell_type in np.unique(cell_labels):
        # 找到source和target中属于该细胞类型的样本
        source_idx = (cell_labels == cell_type)
        target_idx = (cell_labels == cell_type)
        
        # 计算该细胞类型的映射一致性
        consistency = measure_mapping_consistency(
            source[source_idx], 
            target[target_idx],
            transport[source_idx]
        )
        consistency_scores[f"cell_type_{cell_type}"] = consistency
    
    return consistency_scores


def get_module_indices(module_name, data_var=None):
    """
    获取特定模块的基因索引
    Args:
        module_name: 模块名称
        data_var: anndata的var属性，包含gene_module信息
    Returns:
        该模块对应的基因索引列表
    """
    if data_var is not None:
        # 如果提供了data_var，直接从中获取索引
        return np.where(data_var['gene_module'] == module_name)[0]
    else:
        # 否则假设是按顺序排列的模块
        # 这种情况下需要在调用前确保数据已经按模块排序
        raise ValueError("Must provide data_var to get module indices")


def get_time_indices(time_point, data_obs=None):
    """
    获取特定时间点的细胞索引
    Args:
        time_point: 时间点标签
        data_obs: anndata的obs属性，包含time_bin信息
    Returns:
        该时间点对应的细胞索引列表
    """
    if data_obs is not None:
        # 如果提供了data_obs，直接从中获取索引
        return np.where(data_obs['time_bin'] == time_point)[0]
    else:
        raise ValueError("Must provide data_obs to get time indices")


def compute_submatrix_loss(f_or_g, source_subset, target_subset):
    """
    计算子矩阵（特定模块或时间点）的损失
    Args:
        f_or_g: 网络模型（f或g）
        source_subset: 源数据的子集
        target_subset: 目标数据的子集
    Returns:
        子集上的损失值
    """
    # 这里可以根据具体需求定义不同的损失计算方式
    # 例如，可以使用MMD或其他度量
    return losses.compute_scalar_mmd(source_subset, target_subset)


def measure_mapping_consistency(source_cells, target_cells, mapped_cells):
    """
    计算映射的一致性得分
    Args:
        source_cells: 源细胞数据
        target_cells: 目标细胞数据
        mapped_cells: 映射后的细胞数据
    Returns:
        一致性得分
    """
    # 可以使用多种方式计算一致性，这里给出一个示例
    # 例如，可以计算mapped_cells与target_cells的相似度
    consistency = 1.0 - torch.mean(torch.abs(mapped_cells - target_cells))
    return consistency.item()


def get_cell_labels(source, target, cell_label_key):
    """
    获取细胞类型标签
    Args:
        source: 源数据
        target: 目标数据
        cell_label_key: 细胞类型标签的键名
    Returns:
        合并后的细胞类型标签
    """
    if hasattr(source, 'obs') and cell_label_key in source.obs:
        # 如果是AnnData对象
        source_labels = source.obs[cell_label_key]
        target_labels = target.obs[cell_label_key]
    else:
        # 如果是张量，假设标签信息在metadata中
        source_labels = source.metadata[cell_label_key]
        target_labels = target.metadata[cell_label_key]
    
    return np.concatenate([source_labels, target_labels])


def get_module_info(source, target, module_label):
    """
    获取模块信息和权重
    Args:
        source: 源数据
        target: 目标数据
        module_label: 模块标签的键名
    Returns:
        模块权重字典
    """
    if hasattr(source, 'var') and module_label in source.var:
        # 获取所有唯一的模块名称
        modules = np.unique(source.var[module_label])
        # 创建一个简单的权重字典，可以根据需要修改权重分配方式
        module_weights = {module: 1.0 for module in modules}
        return module_weights
    return None


def get_time_info(source, target, time_label):
    """
    获取时间信息和权重
    Args:
        source: 源数据
        target: 目标数据
        time_label: 时间标签的键名
    Returns:
        时间权重字典
    """
    if hasattr(source, 'obs') and time_label in source.obs:
        # 获取所有唯一的时间点
        time_points = np.unique(source.obs[time_label])
        # 创建一个简单的权重字典，可以根据需要修改权重分配方式
        time_weights = {time: 1.0 for time in time_points}
        return time_weights
    return None

"""
关键技术点总结
使用 ICNN (Input Convex Neural Network)：
• 在最优传输的对偶形式中，需要势函数 f(x) 对 x 可微并且是凸函数。ICNN 是一类特殊设计的网络，使得网络输出随输入凸或半凸。
• 通过隐式或显式的约束（比如对权重加非负约束或其他特殊结构），保证网络的凸性。
Kantorovich 对偶：
• 最优传输的 Kantorovich 对偶告诉我们，W2 距离可以用对偶变量 f 来表示，满足一定的不等式约束。
• 具体在代码里，f 用来估计对偶势，g 是运输映射，相当于用深度网络实现"往目标分布搬运"的过程。
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
从高层来讲，它依赖"在对偶理论下，对分布映射进行深度学习"的想法，并通过 ICNN 的凸性为对偶势函数提供模型保证，从而完成对最优传输的学习过程。
"""
