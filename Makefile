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
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt

test:
	. venv/bin/activate && pytest tests/ -v

test-watch:
	. venv/bin/activate && ptw tests/

test-coverage:
	. venv/bin/activate && pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint:
	. venv/bin/activate && flake8 src/ tests/

format:
	. venv/bin/activate && black src/ tests/
	. venv/bin/activate && isort src/ tests/

check-format:
	. venv/bin/activate && black --check src/ tests/
	. venv/bin/activate && isort --check-only src/ tests/

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