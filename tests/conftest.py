import pytest
import time


@pytest.fixture(autouse=True)
def timer_function_scope():
    """Times each test function."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"Time cost: {elapsed:.3f}s")
