import scanpy as sc
import pandas as pd
import numpy as np

# 生成模拟数据
adata = sc.datasets.pbmc3k_processed()
# 增加transport列，随机分配源和目标标签
adata.obs['transport'] = np.random.choice(['source', 'target'], size=adata.shape[0])
# 保存数据
adata.write('test_data.h5ad')
