# Project Structure

## 📁 Directory Layout

```
claude-test/
├── README.md                    # Main project documentation
├── Makefile                     # Build and test automation
├── pyproject.toml              # Python project configuration
├── pytest.ini                 # Test configuration
├── environment.yml             # Conda environment specification
├── requirements.txt            # Python dependencies
├── requirements-minimal.txt    # Minimal dependencies
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── types.py               # Type definitions
│   ├── exceptions.py          # Custom exceptions
│   ├── image_processing.py    # Original implementation
│   ├── image_processing_v2.py # New modular API
│   ├── data_processing.py     # Data processing utilities
│   ├── ml_algorithms.py       # ML algorithm implementations
│   │
│   ├── core/                  # Core processing components
│   │   ├── __init__.py
│   │   ├── interfaces.py      # Abstract interfaces
│   │   ├── validators.py      # Input validation
│   │   ├── transformers.py    # Image transformations
│   │   └── processors.py      # Main processing logic
│   │
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       └── image_utils.py     # Image helper functions
│
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── conftest.py           # Test configuration
│   ├── test_image_processing.py
│   ├── test_data_processing.py
│   └── test_ml_algorithms.py
│
├── scripts/                   # Executable scripts
│   ├── debug_canva_interview.py   # Debug/demo script
│   └── interview_setup.sh         # Environment setup
│
├── benchmarks/               # Performance benchmarking
│   ├── benchmark_image_processing.py
│   ├── benchmark_results.txt
│   └── image_processing_benchmark.png
│
├── docs/                     # Documentation
│   ├── ARCHITECTURE.md       # Architecture documentation
│   ├── CANVA_IMAGE_PROCESSING.md  # Feature documentation
│   ├── INTERVIEW_CHECKLIST.md     # Interview preparation
│   ├── MAKEFILE_DEMO.md           # Makefile usage
│   └── TEST_COMMANDS.md           # Testing guide
│
├── examples/                 # Usage examples
│   ├── __init__.py
│   └── interview_examples.py
│
├── notebooks/               # Jupyter notebooks (if any)
├── data/                   # Test data and datasets
└── htmlcov/               # Coverage reports (generated)
```

## 📋 Directory Purposes

### `/src/` - Source Code
- **Main modules**: Core business logic and APIs
- **`core/`**: Modular processing components using clean architecture
- **`utils/`**: Reusable utility functions

### `/tests/` - Test Suite
- **Unit tests**: Individual component testing
- **Integration tests**: Cross-component testing
- **Test fixtures**: Shared test data and configurations

### `/scripts/` - Executable Scripts
- **Debug scripts**: Interactive debugging and exploration
- **Setup scripts**: Environment and system configuration
- **Utility scripts**: One-off operations and tools

### `/benchmarks/` - Performance Testing
- **Benchmark scripts**: Performance measurement tools
- **Results**: Historical performance data
- **Plots**: Performance visualization

### `/docs/` - Documentation
- **Architecture**: System design and structure
- **API documentation**: Usage guides and examples
- **Interview materials**: Preparation and checklists

### `/examples/` - Usage Examples
- **Demo code**: Practical usage examples
- **Tutorials**: Step-by-step guides
- **Sample applications**: Real-world use cases

## 🎯 Benefits of This Structure

### 1. **Clear Separation of Concerns**
- Source code, tests, scripts, and docs are clearly separated
- Related files are grouped together
- Easy to find specific types of files

### 2. **Professional Standards**
- Follows Python packaging conventions
- Compatible with CI/CD systems
- Standard directory names that tools expect

### 3. **Scalability**
- Easy to add new components
- Clear place for new file types
- Modular structure supports growth

### 4. **Maintainability**
- Dependencies are clear from directory structure
- Easy to understand project organization
- Consistent file placement rules

## 🚀 Navigation Guide

### Common Tasks:

**Running the main function:**
```bash
python -m src.image_processing_v2
```

**Running tests:**
```bash
pytest tests/
```

**Running benchmarks:**
```bash
python benchmarks/benchmark_image_processing.py
```

**Interactive debugging:**
```bash
python scripts/debug_canva_interview.py
```

**Building documentation:**
```bash
make docs
```

### Development Workflow:

1. **Code changes**: Edit files in `/src/`
2. **Add tests**: Create/update files in `/tests/`
3. **Run tests**: `pytest tests/`
4. **Update docs**: Edit files in `/docs/`
5. **Performance check**: Run `/benchmarks/`

This structure provides a clean, professional, and scalable foundation for the project.