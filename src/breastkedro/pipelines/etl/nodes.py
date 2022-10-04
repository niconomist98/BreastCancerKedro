"""
This is a boilerplate pipeline 'etl'
generated using Kedro 0.18.3
"""
import importlib
import logging
from typing import Any, Dict, Tuple

import mlflow
import pandas as pd
import numpy as np

# Assemble pipeline(s)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity
from deepchecks.tabular.suites import train_test_validation

import great_expectations as ge
from great_expectations.checkpoint.types.checkpoint_result import CheckpointResult

from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures

logger = logging.getLogger(__name__)


