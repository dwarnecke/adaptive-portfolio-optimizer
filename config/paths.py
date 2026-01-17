__author__ = "Dylan Warnecke"
__email__ = "dylan.warnecke@gmail.com"

"""
Configuration file for project paths.
"""

from pathlib import Path

# Define project root and data/model directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REGIMES_DIR = OUTPUTS_DIR / "regimes"
DATASETS_DIR = OUTPUTS_DIR / "datasets"
MODELS_DIR = OUTPUTS_DIR / "models"
