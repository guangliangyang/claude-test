# 🧪 Unit Test 运行指南

## 💡 前提条件
```bash
cd /Users/andy/workspace/claude-test
source venv/bin/activate  # 激活虚拟环境
```

## 🚀 基本测试命令

### 1. 运行所有测试
```bash
# 简单运行
pytest

# 详细输出
pytest -v

# 使用Makefile（推荐）
make test
```

### 2. 运行特定测试文件
```bash
# 只测试ML算法
pytest tests/test_ml_algorithms.py -v

# 只测试数据处理
pytest tests/test_data_processing.py -v
```

### 3. 运行特定测试类
```bash
# 只测试线性回归
pytest tests/test_ml_algorithms.py::TestLinearRegression -v

# 只测试数据处理器
pytest tests/test_data_processing.py::TestDataProcessor -v
```

### 4. 运行特定测试方法
```bash
# 测试特定方法
pytest tests/test_ml_algorithms.py::TestLinearRegression::test_fit_and_predict -v

# 测试初始化
pytest tests/test_ml_algorithms.py::TestKMeans::test_initialization -v
```

## 📊 测试覆盖率

### 1. 基本覆盖率报告
```bash
pytest --cov=src

# 显示未覆盖的行
pytest --cov=src --cov-report=term-missing

# 使用Makefile
make test-coverage
```

### 2. HTML覆盖率报告
```bash
pytest --cov=src --cov-report=html
# 然后打开 htmlcov/index.html 查看详细报告
```

### 3. 覆盖率阈值检查
```bash
# 要求最低90%覆盖率
pytest --cov=src --cov-fail-under=90
```

## 🏷️ 测试标记 (Markers)

### 1. 跳过慢速测试
```bash
# 跳过标记为slow的测试
pytest -m "not slow"
```

### 2. 只运行集成测试
```bash
pytest -m integration
```

### 3. 运行特定优先级测试
```bash
# 如果有priority标记
pytest -m "priority_high"
```

## 🐛 调试模式

### 1. 显示print输出
```bash
pytest -s  # --capture=no
```

### 2. 遇到失败就停止
```bash
pytest -x  # --exitfirst
```

### 3. 最多失败N次后停止
```bash
pytest --maxfail=3
```

### 4. 详细失败信息
```bash
pytest --tb=long  # 详细traceback
pytest --tb=short # 简短traceback
pytest --tb=line  # 一行traceback
```

## 🔍 高级测试选项

### 1. 并行测试 (如果安装了pytest-xdist)
```bash
pytest -n auto  # 自动检测CPU核心数
pytest -n 4     # 使用4个进程
```

### 2. 重复运行测试
```bash
pytest --count=10  # 重复10次（需要pytest-repeat）
```

### 3. 随机顺序运行
```bash
pytest --random-order  # 随机顺序（需要pytest-random-order）
```

## 📈 持续集成 (CI) 模式

### 1. CI友好的输出
```bash
pytest --quiet --tb=short --cov=src --cov-report=xml
```

### 2. JUnit XML报告
```bash
pytest --junitxml=test-results.xml
```

## 🛠️ 实际面试演示命令

### 演示1: 快速验证所有功能
```bash
make test
```

### 演示2: 展示测试覆盖率
```bash
make test-coverage
```

### 演示3: 测试特定算法
```bash
pytest tests/test_ml_algorithms.py::TestLinearRegression -v
```

### 演示4: 展示测试驱动开发
```bash
# 显示一个测试的详细执行过程
pytest tests/test_ml_algorithms.py::TestLinearRegression::test_fit_and_predict -v -s
```

## 💡 面试建议

1. **展示测试金字塔**: 单元测试 → 集成测试 → 端到端测试
2. **强调TDD**: 测试先行的开发方式
3. **代码覆盖率**: 展示高质量的测试覆盖
4. **测试分类**: 快速测试 vs 慢速测试的分离
5. **CI/CD就绪**: 展示持续集成友好的测试设置

## ⚡ 快速备忘录

```bash
# 最常用的命令
pytest -v                           # 详细运行所有测试
pytest tests/test_ml_algorithms.py  # 运行ML测试
pytest --cov=src --cov-report=term-missing  # 覆盖率报告
make test                           # 使用Makefile运行
make test-coverage                  # 覆盖率 + 测试
```