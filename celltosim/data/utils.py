"""
数据处理工具函数
"""

from torch.utils.data import DataLoader
from itertools import groupby


def cast_dataset_to_loader(dataset, **kwargs):
    """
    将数据集转换为DataLoader
    
    Args:
        dataset: 数据集对象
        kwargs: DataLoader参数
    Returns:
        DataLoader对象
    """
    batch_size = kwargs.pop('batch_size', 128)
    shuffle = kwargs.pop('shuffle', True)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def cycle(iterable):
    """
    创建无限循环的迭代器
    
    Args:
        iterable: 可迭代对象
    Yields:
        循环的元素
    """
    while True:
        for x in iterable:
            yield x 