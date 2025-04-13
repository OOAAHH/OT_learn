from setuptools import setup, find_packages

setup(
    name="celltosim",
    version="0.1.0",
    description="简化版的基于最优传输的单细胞数据分析工具",
    author="CellOT Team",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "pandas>=1.1.0",
        "anndata>=0.7.6",
        "scikit-learn>=0.24.0",
        "tqdm>=4.50.0",
        "tensorboard>=2.4.0",
    ],
    python_requires=">=3.7",
) 