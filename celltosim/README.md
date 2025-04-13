# CellToSim

基于最优传输理论的单细胞数据分析简化工具

## 简介

CellToSim 是一个基于最优传输（Optimal Transport）理论的单细胞数据分析工具，可以学习并预测细胞状态的转变。本工具是基于原始 CellOT 工具的简化版，只保留了核心的最优传输功能，并进行了代码重构和简化。

### 核心特点

- 使用 Input Convex Neural Network (ICNN) 实现最优传输
- 保留源代码中最有价值的最优传输算法实现
- 简化接口和数据处理流程
- 提供易用的训练和推理脚本

## 安装

### 依赖项

- Python 3.7+
- PyTorch 1.7+
- anndata
- numpy
- scipy
- pandas
- scikit-learn
- tqdm
- tensorboard

### 安装步骤

```bash
git clone https://github.com/username/celltosim.git
cd celltosim
pip install -e .
```

## 使用方法

### 数据格式

CellToSim 使用 AnnData 格式的单细胞数据。数据中需要包含一个表示传输标签的列（默认为 `transport`），用于区分源细胞和目标细胞。

### 训练模型

```bash
python -m celltosim.examples.train_cellot \
    --data path/to/anndata.h5ad \
    --source source_label \
    --target target_label \
    --output output_dir \
    --n-iters 5000
```

### 使用模型进行预测

```bash
python -m celltosim.examples.predict_cellot \
    --data path/to/anndata.h5ad \
    --source source_label \
    --model path/to/model.pt \
    --output predicted_cells.h5ad
```

## 实现细节

### 最优传输

CellToSim 使用两个 ICNN 网络来实现最优传输：

1. `f` 网络：表示 Kantorovich 势函数
2. `g` 网络：计算从源分布到目标分布的传输映射

通过这两个网络的联合训练，实现了细胞状态转变的预测。

### 网络架构

- 使用 Input Convex Neural Networks (ICNNs) 确保凸性
- 通过非负权重和单调激活函数实现
- 支持权重正则化和约束

## 引用

如果您在研究中使用了 CellToSim，请引用原始的 CellOT 论文：

```
@article{cellot2021,
  title={CellOT: Predicting cellular responses to perturbations with optimal transport},
  author={Original Authors},
  journal={Original Journal},
  year={2021}
}
```

## 许可证

本项目使用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。 