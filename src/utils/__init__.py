# Import utility functions for easier access in the utils package

from .data_processing import load_data

# Define the public API for the utils package
__all__ = ["load_data"]
