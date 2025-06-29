# 🎯 Canva AI Engineer 面试检查清单

## 📋 面试前准备 (5分钟)

### ✅ 环境检查
```bash
cd /Users/andy/workspace/claude-test
./interview_setup.sh
```

### ✅ 核心功能验证
```bash
# 运行快速测试确保一切正常
make test
# 运行演示示例
python examples/interview_examples.py
```

---

## 🚀 面试期间可展示的内容

### 1. 🤖 机器学习算法实现
**文件**: `src/ml_algorithms.py`

**可展示的算法**:
- ✅ **线性回归** (梯度下降实现)
- ✅ **K-Means聚类** (完整实现)  
- ✅ **朴素贝叶斯分类器** (兼容sklearn API)
- ✅ **工具函数** (train_test_split, 评估指标)

**演示命令**:
```python
from src.ml_algorithms import LinearRegression, KMeans, NaiveBayes
# 所有算法都有完整的测试覆盖
```

### 2. 📊 数据预处理能力
**文件**: `src/data_processing.py`

**可展示的功能**:
- ✅ **缺失值处理** (多种策略)
- ✅ **分类变量编码** (one-hot, label encoding)
- ✅ **特征缩放** (标准化, 归一化)
- ✅ **异常值检测** (IQR, Z-score)
- ✅ **数据集生成** (回归、分类、时间序列)

### 3. 🧪 测试驱动开发
**文件**: `tests/`

**可展示的测试**:
- ✅ **单元测试** (35个测试用例)
- ✅ **集成测试** (完整流水线)
- ✅ **覆盖率报告** (pytest-cov)
- ✅ **性能测试** (与sklearn对比)

**演示命令**:
```bash
pytest tests/test_ml_algorithms.py::TestLinearRegression -v
```

### 4. 📝 代码质量
**配置文件**: `.flake8`, `pyproject.toml`, `Makefile`

**可展示的工具**:
- ✅ **代码格式化** (Black)
- ✅ **静态分析** (Flake8)
- ✅ **导入排序** (isort)
- ✅ **自动化构建** (Make)

---

## 💡 面试策略建议

### 🎪 展示顺序建议
1. **快速环境演示** (1分钟)
   ```bash
   ./interview_setup.sh
   ```

2. **核心算法讲解** (15分钟)
   - 选择一个算法详细讲解实现
   - 展示测试用例
   - 运行实际示例

3. **数据处理流水线** (10分钟)
   ```python
   python examples/interview_examples.py
   ```

4. **代码质量展示** (5分钟)
   ```bash
   make lint
   make test-coverage
   ```

### 🗣️ 可以强调的技能点
- ✅ **算法理解**: 从零实现经典ML算法
- ✅ **代码质量**: 完整的测试覆盖率
- ✅ **工程实践**: 配置管理、自动化工具
- ✅ **API设计**: 兼容sklearn的接口设计
- ✅ **性能对比**: 与标准库的对比验证

### 🔧 常见面试问题准备
1. **"实现一个线性回归算法"**
   - ✅ 已实现: `LinearRegression` 类
   - ✅ 包含梯度下降优化
   - ✅ 有完整测试验证

2. **"处理带缺失值的数据集"**
   - ✅ 已实现: `DataProcessor.handle_missing_values()`
   - ✅ 支持多种填充策略
   - ✅ 有实际数据演示

3. **"如何评估模型性能"**
   - ✅ 已实现: 多种评估指标
   - ✅ 交叉验证框架
   - ✅ 与sklearn对比验证

4. **"代码质量如何保证"**
   - ✅ 单元测试 (pytest)
   - ✅ 代码格式化 (black)
   - ✅ 静态分析 (flake8)
   - ✅ CI/CD ready

---

## ⚡ 快速参考命令

### 测试相关
```bash
make test                    # 运行所有测试
make test-coverage          # 测试 + 覆盖率
pytest tests/test_ml_algorithms.py -v  # 单个文件测试
```

### 代码质量
```bash
make lint                   # 代码检查
make format                # 代码格式化
make check-format          # 检查格式
```

### 演示示例
```bash
python examples/interview_examples.py    # 所有示例
jupyter notebook                         # 交互式环境
```

---

## 🎯 最后检查

**面试开始前30秒**:
- [ ] 终端已打开到项目目录
- [ ] 虚拟环境已激活 (`source venv/bin/activate`)
- [ ] IDE/编辑器已打开相关文件
- [ ] 网络连接正常 (如需要安装依赖)

**心理准备**:
- [ ] 自信展示自己的代码实现
- [ ] 准备解释算法原理和设计决策
- [ ] 可以现场修改代码或添加功能
- [ ] 强调测试驱动开发和代码质量

## 🍀 祝您面试成功！

记住: 这个环境展示了您的**完整工程能力**，不仅仅是算法实现，还包括测试、文档、代码质量等软件工程的各个方面。