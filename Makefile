.PHONY: help install test lint format clean run-tests check-format

help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  test         - Run all tests"
	@echo "  test-watch   - Run tests in watch mode"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  check-format - Check if code is properly formatted"
	@echo "  clean        - Remove cache files and test artifacts"
	@echo "  run-tests    - Run tests with coverage report"

install:
	conda env create -f environment.yml || conda env update -f environment.yml

test:
	conda run -n ml-env pytest tests/ -v

test-watch:
	conda run -n ml-env ptw tests/

test-coverage:
	conda run -n ml-env pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint:
	conda run -n ml-env flake8 src/ tests/

format:
	conda run -n ml-env black src/ tests/
	conda run -n ml-env isort src/ tests/

check-format:
	conda run -n ml-env black --check src/ tests/
	conda run -n ml-env isort --check-only src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

run-tests: lint test-coverage
	@echo "All checks passed!"