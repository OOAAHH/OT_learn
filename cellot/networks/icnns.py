"""
Input Convex Neural Networks (ICNNs) 实现

主要思路:
1. ICNNs是一类特殊的神经网络,保证了输出相对于输入是凸函数
2. 网络结构通过以下方式实现凸性:
   - 所有权重矩阵W必须非负
   - 使用单调递增的激活函数(如ReLU、LeakyReLU)
3. 主要用途:
   - 优化问题求解
   - 最优传输问题
   - 生成对抗网络
4. 本实现包含:
   - 非负线性层实现
   - ICNN主网络结构
   - 凸性测试函数
"""

import torch
from torch import autograd
import numpy as np
from torch import nn 
from numpy.testing import assert_allclose

# 支持的激活函数字典
ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
}

# 非负线性层实现
class NonNegativeLinear(nn.Linear):
    """实现权重矩阵必须非负的线性层"""
    def __init__(self, *args, beta=1.0, **kwargs):
        super(NonNegativeLinear, self).__init__(*args, **kwargs)
        self.beta = beta  # softplus函数的beta参数
        return

    def forward(self, x):
        return nn.functional.linear(x, self.kernel(), self.bias)

    def kernel(self):
        # 使用softplus确保权重非负
        return nn.functional.softplus(self.weight, beta=self.beta)

# ICNN主网络结构
class ICNN(nn.Module):
    """Input Convex Neural Network实现"""
    def __init__(
        self,
        input_dim,          # 输入维度
        hidden_units,       # 隐藏层单元数列表
        activation="LeakyReLU",  # 激活函数
        softplus_W_kernels=False,  # 是否使用softplus进行权重约束
        softplus_beta=1,    # softplus的beta参数
        std=0.1,           # 初始化标准差
        fnorm_penalty=0,    # Frobenius范数惩罚项
        kernel_init_fxn=None,  # 权重初始化函数
    ):
        super(ICNN, self).__init__()
        self.fnorm_penalty = fnorm_penalty
        self.softplus_W_kernels = softplus_W_kernels

        # 设置激活函数
        if isinstance(activation, str):
            activation = ACTIVATIONS[activation.lower().replace("_", "")]
        self.sigma = activation

        units = hidden_units + [1]

        # 构建网络层
        # z_{l+1} = \sigma_l(W_l*z_l + A_l*x + b_l)
        if self.softplus_W_kernels:
            def WLinear(*args, **kwargs):
                return NonNegativeLinear(*args, **kwargs, beta=softplus_beta)
        else:
            WLinear = nn.Linear

        # W矩阵列表(层间连接)
        self.W = nn.ModuleList(
            [
                WLinear(idim, odim, bias=False)
                for idim, odim in zip(units[:-1], units[1:])
            ]
        )

        # A矩阵列表(输入连接)
        self.A = nn.ModuleList(
            [nn.Linear(input_dim, odim, bias=True) for odim in units]
        )

        # 初始化权重
        if kernel_init_fxn is not None:
            for layer in self.A:
                kernel_init_fxn(layer.weight)
                nn.init.zeros_(layer.bias)
            for layer in self.W:
                kernel_init_fxn(layer.weight)
        return

    def forward(self, x):
        """前向传播"""
        # 第一层特殊处理
        z = self.sigma(0.2)(self.A[0](x))
        z = z * z

        # 中间层
        for W, A in zip(self.W[:-1], self.A[1:-1]):
            z = self.sigma(0.2)(W(z) + A(x))

        # 输出层
        y = self.W[-1](z) + self.A[-1](x)
        return y

    def transport(self, x):
        """计算最优传输映射"""
        assert x.requires_grad
        # 计算对输入的梯度
        (output,) = autograd.grad(
            self.forward(x),
            x,
            create_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((x.size()[0], 1), device=x.device).float(),
        )
        return output

    def clamp_w(self):
        """将W矩阵的权重截断为非负"""
        if self.softplus_W_kernels:
            return
        for w in self.W:
            w.weight.data = w.weight.data.clamp(min=0)
        return

    def penalize_w(self):
        """计算负权重的惩罚项"""
        return self.fnorm_penalty * sum(
            map(lambda x: torch.nn.functional.relu(-x.weight).norm(), self.W)
        )

# 测试ICNN的凸性
def test_icnn_convexity(icnn):
    """
    测试ICNN是否满足凸性质
    对任意两点x,y和t∈[0,1],验证:
    f(tx + (1-t)y) ≤ tf(x) + (1-t)f(y)
    """
    data_dim = icnn.A[0].in_features

    zeros = np.zeros(100)
    for _ in range(100):
        x = torch.rand((100, data_dim))
        y = torch.rand((100, data_dim))

        fx = icnn(x)
        fy = icnn(y)

        for t in np.linspace(0, 1, 10):
            fxy = icnn(t * x + (1 - t) * y)
            res = (t * fx + (1 - t) * fy) - fxy
            res = res.detach().numpy().squeeze()
            assert_allclose(np.minimum(res, 0), zeros, atol=1e-6)
