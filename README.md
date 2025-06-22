# Canva AI Engineer Interview Preparation Environment

This is a comprehensive testing environment prepared for AI-assisted programming interviews, specifically designed for the Canva AI Engineer position.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup
1. **Clone/Navigate to the project directory**
   ```bash
   cd /Users/andy/workspace/claude-test
   ```

2. **Install dependencies**
   ```bash
   make install
   # OR manually:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Activate the virtual environment**
   ```bash
   source venv/bin/activate
   ```

4. **Run tests to verify setup**
   ```bash
   make test
   # OR
   pytest tests/ -v
   ```

## 📁 Project Structure

```
claude-test/
├── src/                     # Source code
│   ├── __init__.py
│   ├── ml_algorithms.py     # Machine learning implementations
│   └── data_processing.py   # Data preprocessing utilities
├── tests/                   # Test files
│   ├── __init__.py
│   ├── test_ml_algorithms.py
│   └── test_data_processing.py
├── data/                    # Data directory (for datasets)
├── notebooks/               # Jupyter notebooks for exploration
├── venv/                    # Virtual environment
├── requirements.txt         # Python dependencies
├── pytest.ini             # Pytest configuration
├── pyproject.toml          # Black/isort configuration
├── .flake8                 # Flake8 linting configuration
├── Makefile                # Build automation
└── README.md               # This file
```

## 🛠️ Available Tools & Libraries

### Machine Learning Libraries
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning toolkit
- **SciPy**: Scientific computing
- **PyTorch**: Deep learning framework
- **TensorFlow**: Deep learning platform

### Data Visualization
- **Matplotlib**: Basic plotting
- **Seaborn**: Statistical visualization
- **Plotly**: Interactive plots

### Development Tools
- **Pytest**: Testing framework with coverage
- **Black**: Code formatting
- **Flake8**: Code linting
- **isort**: Import sorting
- **Jupyter**: Interactive development

## 🧪 Testing

### Run All Tests
```bash
make test
# OR
pytest tests/ -v
```

### Run Tests with Coverage
```bash
make test-coverage
# OR
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
```

### Run Tests in Watch Mode (for development)
```bash
make test-watch
# OR
ptw tests/
```

### Run Specific Test Files
```bash
pytest tests/test_ml_algorithms.py -v
pytest tests/test_data_processing.py -v
```

### Run Tests by Markers
```bash
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Only integration tests
```

## 🔧 Code Quality

### Format Code
```bash
make format
# OR manually:
black src/ tests/
isort src/ tests/
```

### Check Code Style
```bash
make lint
# OR
flake8 src/ tests/
```

### Check if Code is Properly Formatted
```bash
make check-format
```

## 🤖 Implemented AI Algorithms

### Machine Learning Algorithms (`src/ml_algorithms.py`)

1. **Linear Regression**
   - Custom implementation with gradient descent
   - Configurable learning rate and iterations
   - Cost history tracking

2. **K-Means Clustering**
   - Unsupervised clustering algorithm
   - Configurable number of clusters
   - Random initialization with seed support

3. **Naive Bayes Classifier**
   - Gaussian Naive Bayes implementation
   - Probability prediction support
   - Compatible with scikit-learn API

4. **Utility Functions**
   - Train/test split
   - Accuracy scoring
   - Mean squared error
   - Mean absolute error

### Data Processing (`src/data_processing.py`)

1. **DataProcessor Class**
   - Missing value handling (mean, median, mode, forward/backward fill)
   - Categorical encoding (one-hot, label encoding)
   - Feature scaling (standard, min-max)
   - Outlier removal (IQR, z-score methods)

2. **Dataset Generation**
   - Sample regression datasets
   - Classification datasets
   - Time series generation

## 💡 Usage Examples

### Quick ML Pipeline Example
```python
from src.ml_algorithms import LinearRegression, train_test_split, mean_squared_error
from src.data_processing import create_sample_dataset

# Create dataset
X, y = create_sample_dataset(n_samples=1000, n_features=5, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
```

### Data Processing Example
```python
import pandas as pd
from src.data_processing import DataProcessor

# Create sample data
df = pd.DataFrame({
    'age': [25, 30, None, 40, 35],
    'category': ['A', 'B', 'A', 'C', 'B'],
    'income': [50000, 60000, 55000, 80000, None]
})

# Process data
processor = DataProcessor()

# Handle missing values
df = processor.handle_missing_values(df, strategy='mean', columns=['age', 'income'])

# Encode categorical variables
df = processor.encode_categorical(df, columns=['category'], method='onehot')

# Scale features
df = processor.scale_features(df, columns=['age', 'income'], method='standard')
```

## 🎯 Interview Preparation Tips

### Common AI/ML Interview Topics Covered:
1. **Supervised Learning**: Linear Regression, Naive Bayes
2. **Unsupervised Learning**: K-Means Clustering
3. **Data Preprocessing**: Missing values, encoding, scaling, outliers
4. **Model Evaluation**: Train/test split, accuracy, MSE, MAE
5. **Code Quality**: Unit testing, documentation, clean code practices

### Best Practices Demonstrated:
- ✅ Comprehensive unit testing with pytest
- ✅ Code formatting with Black
- ✅ Type hints for better code documentation
- ✅ Error handling and input validation
- ✅ Modular design and separation of concerns
- ✅ Configuration management
- ✅ Reproducible results with random seeds

## 🚨 Troubleshooting

### Virtual Environment Issues
```bash
# If virtual environment activation fails:
deactivate  # Exit current environment
rm -rf venv  # Remove old environment
make install  # Reinstall
```

### Import Errors
```bash
# Make sure you're in the virtual environment:
source venv/bin/activate

# And that you're in the correct directory:
pwd  # Should show /Users/andy/workspace/claude-test
```

### Test Failures
```bash
# Run tests with more verbose output:
pytest tests/ -v -s

# Run a specific test:
pytest tests/test_ml_algorithms.py::TestLinearRegression::test_fit_and_predict -v
```

## 🔄 Continuous Integration Ready

The project is set up for easy CI/CD integration with:
- Automated testing with pytest
- Code quality checks with flake8
- Code formatting verification with black
- Coverage reporting
- Make targets for all operations

## 📝 Notes for Interview

1. **Environment is fully configured** - All dependencies installed and tested
2. **Multiple algorithms implemented** - Ready for various problem types
3. **Comprehensive test suite** - Demonstrates testing best practices
4. **Clean code structure** - Easy to navigate and extend
5. **Documentation** - Well-documented code and usage examples

Good luck with your Canva AI Engineer interview! 🎉