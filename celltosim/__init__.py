"""
CellToSim - 基于最优传输的简化单细胞预测工具

核心功能:
1. 使用Input Convex Neural Network (ICNN)实现最优传输
2. 训练源分布到目标分布的映射模型
3. 预测细胞状态变化
"""

from . import models
from . import networks
from . import data
from . import train
from . import utils 