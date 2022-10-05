"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.2
"""
# Importing necessary libraries.

import importlib
import logging
from typing import Any, Dict, Tuple
from sklearn import preprocessing
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

def etl_processing(data: pd.DataFrame) -> pd.DataFrame:
    """
    General transformations to the data like removing columns with
    the same constant value, duplicated columns., duplicate values
    Args:
        data: raw data after extract
        parameters: list of the general transforms to apply to all the data
    Returns:
        pd.DataFrame: transformed data
"""
    mlflow.set_experiment('BreastCancer')
    mlflow.log_param("shape raw_data", data.shape)

    data = (data
                    .pipe(drop_exact_duplicates)
                    .pipe(drop_duplicates, drop_cols=['index'])
                    .pipe(clean_misslabeled)
                    .pipe(transform_output)
                    .pipe(drop_exact_duplicates)
                    .pipe(sort_data, col = 'diagnosis')
                    )

    pipe_functions = [
        ('drop_constant_values', DropConstantFeatures(tol=1, missing_values='ignore')),
        ('drop_duplicates', DropDuplicateFeatures(missing_values='ignore'))
    ]

    # get methods name for experimentation tracking
    methods = []
    for name, _ in pipe_functions:
        methods.append(name)

    mlflow.set_experiment('BreastCancer')
    mlflow.log_param('etl_transforms', methods)

    pipeline_train_data = Pipeline(steps=pipe_functions)
    # apply transformation to data
    data_transformed = pipeline_train_data.fit_transform(data)

    mlflow.log_param("shape data etl", data_transformed.shape)

    return data_transformed


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.
    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_processing.yml.
    Returns:
        Split data.
    """
    mlflow.set_experiment('BreastCancer')
    mlflow.log_param("split random_state", parameters['split']['random_state'])
    mlflow.log_param("split test_size", parameters['split']['test_size'])

    # defining features and target_column
    x_features = data[parameters['features']]
    y_target = data[parameters['target_column']]

    x_train, x_test, y_train, y_test = train_test_split(
        x_features,
        y_target,
        test_size=parameters['split']['test_size'],
        random_state=parameters['split']['random_state']
    )

    mlflow.log_param(f"shape train", x_train.shape)
    mlflow.log_param(f"shape test", x_test.shape)

    return x_train, x_test, y_train, y_test



# Function to validate data integrity
def data_integrity_validation(data: pd.DataFrame,
                              parameters: Dict) -> pd.DataFrame:

    numerical_features = parameters['categorical_cols']
    label = parameters['target_column']

    dataset = Dataset(data,
                 label=label,
                numerical_features=numerical_features)

    # Run Suite:
    integ_suite = data_integrity()
    suite_result = integ_suite.run(dataset)
    mlflow.set_experiment('readmission')
    mlflow.log_param(f"data integrity validation", str(suite_result.passed()))
    if not suite_result.passed():
        # save report in data/08_reporting
        suite_result.save_as_html('data/08_reporting/data_integrity_check.html')
        logger.error("data integrity not pass validation tests")
        #raise Exception("data integrity not pass validation tests")
    return data


# function to print shape of the dataframe
def print_shape(data: pd.DataFrame, msg: str = 'Shape =') -> pd.DataFrame:
    """Print shape of dataframe."""
    print(f'{data.shape}{msg}')
    return data

#function to sort data
def sort_data(data: pd.DataFrame, col: str) -> pd.DataFrame:
    "Sort data by and specific column"
    data = data.sort_values(by=col,ascending= False)
    return data


# remove duplicates from data based on a column
def drop_exact_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    data=data.drop_duplicates(keep='first')
    return data


# remove duplicates from data based on a column
def drop_duplicates(data: pd.DataFrame,
                    drop_cols: list) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    data = data.drop_duplicates(subset=drop_cols, keep='first')
    return data

# function to transform output
def transform_output(data: pd.DataFrame) -> pd.DataFrame:
    """ Replace target column to 1 and 0 values"""
    label_encoding = preprocessing.LabelEncoder()
    data['diagnosis'] = label_encoding.fit_transform(data['diagnosis'])
    return data

#Function to clean missing values from the dataset
def clean_misslabeled(data:pd.DataFrame)->pd.DataFrame():
    """function to drop a row which target is wrongly labeled  """
    data = data[(data['diagnosis'] == 'B') | (data['diagnosis'] == 'M')]
    return data
