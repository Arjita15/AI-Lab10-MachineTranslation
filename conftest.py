"""Lets `pytest` (and direct script runs) find the `src` package without needing PYTHONPATH set manually."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
