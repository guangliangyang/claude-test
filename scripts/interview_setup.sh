#!/bin/bash

# Canva AI Engineer Interview Setup Script
# 快速启动面试环境

echo "🚀 Canva AI Engineer 面试环境启动"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ 错误: 请在项目根目录运行此脚本"
    exit 1
fi

# Activate virtual environment
echo "📦 激活虚拟环境..."
source venv/bin/activate

# Quick test
echo "🧪 快速测试环境..."
python -c "
import numpy as np
import pandas as pd
import sklearn
from src.ml_algorithms import LinearRegression, KMeans, NaiveBayes
print('✅ 所有依赖库正常加载')
print(f'✅ NumPy版本: {np.__version__}')
print(f'✅ Pandas版本: {pd.__version__}')
print(f'✅ Scikit-learn版本: {sklearn.__version__}')
"

echo ""
echo "🎯 面试环境已就绪！"
echo ""
echo "可用命令:"
echo "  make test          - 运行所有测试"
echo "  make test-coverage - 运行测试并生成覆盖率报告"
echo "  make lint          - 代码质量检查"
echo "  make format        - 代码格式化"
echo ""
echo "示例代码:"
echo "  python examples/interview_examples.py  - 运行所有演示示例"
echo "  jupyter notebook                       - 启动Jupyter环境"
echo ""
echo "核心模块:"
echo "  src/ml_algorithms.py     - 机器学习算法实现"
echo "  src/data_processing.py   - 数据预处理工具"
echo "  tests/                   - 完整测试套件"
echo ""
echo "💡 面试提示: 所有算法都有完整的单元测试，可以展示测试驱动开发"
echo "🔥 祝您面试顺利！"