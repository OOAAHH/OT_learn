# CellToSim 深度学习教材使用指南

## 教材概述

本教材以CellToSim项目为核心，通过"动手深度学习"的方式，帮助生物信息学专业的学生掌握如何将深度学习技术应用于单细胞组学数据分析。教材采用项目驱动的教学方法，让学生在解决实际生物学问题的过程中学习深度学习的理论和实践。

## 教材结构

### 📚 理论教学部分
```
教学文档/
├── 00_教材总体设计.md           # 教学理念和课程安排
├── 01_摘要_项目背景与核心贡献.md    # 第一部分：背景介绍
├── 02_结果_实验设计与性能评估.md    # 第二部分：实验方法
├── 03_材料和方法_核心技术实现.md    # 第三部分：技术实现
└── 04_讨论和展望_方法改进与应用拓展.md # 第四部分：拓展思考
```

### 🛠️ 实践指南部分
```
教学文档/实践指南/
├── 01_环境配置与数据准备.md       # 环境搭建和数据处理
├── 02_基础模型训练.md           # 基础训练流程
├── 03_多模态损失函数实现.md       # 高级功能实现
└── 04_结果分析与可视化.md        # 结果分析方法
```

## 使用方式

### 👨‍🏫 教师使用指南

#### 课程准备
1. **环境准备**
   ```bash
   # 为学生准备统一的计算环境
   conda create -n celltosim_class python=3.8
   conda activate celltosim_class
   pip install -r requirements.txt
   ```

2. **数据准备**
   ```bash
   # 生成教学用的模拟数据
   python 教学文档/实践指南/generate_synthetic_data.py
   ```

3. **课件准备**
   - 每个章节都包含PPT素材
   - 提供Jupyter Notebook版本的交互式教学
   - 准备演示用的代码示例

#### 教学建议

**第1-2周：项目背景和问题定义**
- 使用`01_摘要_项目背景与核心贡献.md`
- 重点：生物学背景、技术挑战、创新点
- 互动：让学生思考传统方法的局限性
- 作业：阅读相关论文，总结跨物种分析的重要性

**第3-4周：实验设计和评估方法**
- 使用`02_结果_实验设计与性能评估.md`
- 重点：科学实验设计、评估指标、结果解释
- 实践：设计消融实验，分析实验结果
- 作业：设计一个新的评估指标

**第5-10周：核心技术实现**
- 使用`03_材料和方法_核心技术实现.md`和实践指南
- 重点：ICNN原理、多模态损失函数、训练策略
- 实践：逐步实现完整的模型
- 项目：独立实现一个简化版本

**第11-12周：方法改进和应用拓展**
- 使用`04_讨论和展望_方法改进与应用拓展.md`
- 重点：批判性思维、创新能力、应用拓展
- 讨论：方法局限性和改进方案
- 项目：提出创新的改进方案

#### 评估方案
```python
# 评估权重分配
evaluation_weights = {
    "课堂参与": 0.10,
    "作业完成": 0.20,
    "代码实现": 0.30,
    "期中项目": 0.20,
    "期末展示": 0.20
}
```

### 👨‍🎓 学生使用指南

#### 学习准备
1. **预备知识检查**
   - [ ] Python编程基础
   - [ ] 线性代数和概率论
   - [ ] 机器学习基础概念
   - [ ] 生物学基础知识

2. **环境配置**
   ```bash
   # 按照实践指南配置环境
   cd 教学文档/实践指南/
   python test_installation.py
   ```

#### 学习路径

**🔰 初学者路径（本科生推荐）**
1. 仔细阅读每个章节的理论部分
2. 完成所有思考题和概念检查
3. 跟随实践指南完成代码实现
4. 运行示例实验并分析结果
5. 完成章节末尾的练习题

**🔥 进阶路径（研究生推荐）**
1. 深入理解数学原理和算法细节
2. 独立实现核心算法
3. 设计和执行消融实验
4. 提出改进方案并实现
5. 撰写技术报告

**🚀 高级路径（博士生推荐）**
1. 批判性分析现有方法
2. 提出创新的解决方案
3. 扩展到新的应用场景
4. 撰写学术论文
5. 开发新的工具包

#### 学习建议

**时间安排**
- 每周投入8-12小时
- 理论学习：40%
- 代码实践：40%
- 思考讨论：20%

**学习方法**
- 先理解生物学问题，再学习技术方案
- 边学边做，理论与实践结合
- 多思考"为什么"，培养批判性思维
- 积极参与讨论，分享学习心得

**常见困难及解决方案**
```python
common_issues = {
    "数学基础不足": {
        "解决方案": "补充线性代数和最优化理论",
        "推荐资源": ["线性代数应该这样学", "凸优化教程"]
    },
    "编程能力不够": {
        "解决方案": "加强Python和PyTorch练习",
        "推荐资源": ["Python科学计算", "PyTorch官方教程"]
    },
    "生物学背景缺乏": {
        "解决方案": "补充单细胞生物学知识",
        "推荐资源": ["单细胞分析综述", "生物信息学导论"]
    }
}
```

## 教学资源

### 📖 参考资料

#### 理论基础
- Villani, C. "Optimal Transport: Old and New"
- Amos, B. et al. "Input Convex Neural Networks"
- Schiebinger, G. et al. "Optimal-Transport Analysis of Single-Cell Gene Expression"

#### 技术实现
- PyTorch官方文档
- AnnData使用指南
- Scanpy教程

#### 生物学背景
- 单细胞RNA测序技术综述
- 跨物种比较基因组学
- 发育生物学基础

### 💻 代码资源

#### 完整项目代码
```bash
# 克隆完整项目
git clone https://github.com/username/celltosim
cd celltosim
```

#### 教学版本代码
```bash
# 简化的教学版本
git clone https://github.com/username/celltosim-tutorial
cd celltosim-tutorial
```

#### Jupyter Notebook
- 交互式教学笔记本
- 逐步实现的代码示例
- 可视化结果展示

### 📊 数据资源

#### 模拟数据集
- 跨物种单细胞数据
- 时间序列发育数据
- 多样本多模态数据

#### 真实数据集
- 人类PBMC数据
- 小鼠发育数据
- 同源基因映射文件

## 技术支持

### 🔧 环境问题
```bash
# 常见问题解决
# 1. CUDA版本不匹配
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 2. 内存不足
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 3. 依赖冲突
conda env create -f environment.yml
```

### 📞 获取帮助

#### 在线资源
- 项目GitHub Issues
- 课程讨论论坛
- 技术交流群

#### 联系方式
- 邮箱：course-support@example.com
- 办公时间：周二、周四 14:00-16:00
- 在线答疑：每周五 19:00-20:00

## 贡献指南

### 🤝 如何贡献

#### 教师贡献
- 提供教学反馈
- 分享教学经验
- 改进教学内容
- 开发新的练习题

#### 学生贡献
- 报告错误和问题
- 提出改进建议
- 分享学习心得
- 开发扩展功能

#### 贡献流程
1. Fork项目仓库
2. 创建功能分支
3. 提交改进内容
4. 发起Pull Request
5. 代码审查和合并

### 📝 内容更新

#### 版本管理
- 主版本：重大内容更新
- 次版本：新增章节或功能
- 修订版：错误修复和小改进

#### 更新日志
```markdown
## v1.0.0 (2024-01-01)
- 初始版本发布
- 完整的四部分教学内容
- 实践指南和代码示例

## v1.1.0 (2024-02-01)
- 新增高级应用场景
- 改进可视化工具
- 增加更多练习题
```

## 许可证

本教材采用 [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 许可证。

- ✅ 允许：使用、修改、分发、商业使用
- ❗ 要求：署名、相同方式共享
- ❌ 禁止：移除版权信息

## 致谢

感谢所有为本教材开发做出贡献的教师、学生和研究人员。特别感谢：

- 原始CellOT方法的开发者
- 单细胞数据分析社区
- PyTorch和相关开源项目
- 参与测试和反馈的师生

---

**开始学习**: 从 `01_摘要_项目背景与核心贡献.md` 开始你的深度学习之旅！

**问题反馈**: 如有任何问题或建议，请通过GitHub Issues或邮件联系我们。 