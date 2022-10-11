"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.2
"""
import logging
from typing import Any, Dict, Tuple

import mlflow
import pandas as pd
import numpy as np

# Assemble pipeline(s)
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

logger = logging.getLogger(__name__)


def pre_processing(x: pd.DataFrame,
                   y:pd.Series,
                   parameters: Dict[str, Any]) -> pd.DataFrame:
    """data processing only in the train data but not in the test data
    Args:
        data: Data train frame containing features.
    Returns:
        data: Processed data for training .
    """
    data = pd.concat([x,y],axis=1)

    cols_to_keep=['radius_worst', 'concavity_worst', 'fractal_dimension_worst',
       'texture_worst', 'smoothness_worst', 'symmetry_worst', 'perimeter_se',
       'smoothness_se', 'area_se', 'texture_se', 'fractal_dimension_se',
       'symmetry_se', 'diagnosis']
    
    pipe_functions =[
                ('Drop unnecesary columns',FunctionTransformer(drop_cols))]

    methods = ['Drop unnecesary columns']
    pipeline_pre_processing = Pipeline(steps=pipe_functions)
    data_processed = pipeline_pre_processing.fit_transform(data)                                            
    mlflow.set_experiment('BreastCancer')
    mlflow.log_param('pre-processing', methods)

    x_out = data_processed[parameters['featuresfe']]
    y_out = data_processed[parameters['target_columnfe']]

    logger.info(f"Shape = {x_out.shape} pre_processing")

    return x_out, y_out



# function to filter data from a column if is in a list of values
def drop_cols(data: pd.DataFrame) -> pd.DataFrame:
    """Filter columns from data."""
    cols_to_keep=['radius_worst', 'concavity_worst', 'fractal_dimension_worst',
       'texture_worst', 'smoothness_worst', 'symmetry_worst', 'perimeter_se',
       'smoothness_se', 'area_se', 'texture_se', 'fractal_dimension_se',
       'symmetry_se', 'diagnosis']
    data = data[cols_to_keep]
    return data

def first_processing(data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    create pipeline of General transformations to the data like creating new features.
    Args:
        data: train data after splitting
        parameters: list of the general transforms to apply to all the data
    Returns:
        pd.DataFrame: transformed data
    """
    logger.info(f"Shape = {data.shape} first_processing")


    pipe_functions = [
        ('Drop_hc_features',FunctionTransformer(drop_hc_cols)),
       ]


    # get methods name for experimentation tracking
    methods = []
    for name, _ in pipe_functions:
        methods.append(name)
    mlflow.set_experiment('Breast cancer')
    mlflow.log_param('first-processing', methods)

    pipeline_train_data = Pipeline(steps=pipe_functions)
    return data, ('first_processing', pipeline_train_data)


def data_type_split(data: pd.DataFrame, parameters: Dict[str, Any]):

    if parameters['numerical_cols'] and parameters['categorical_cols']:
        numerical_cols = parameters['numerical_cols']
        categorical_cols = parameters['categorical_cols']
    else:
        numerical_cols = make_column_selector(dtype_include=np.number)(data)
        categorical_cols = make_column_selector(dtype_exclude=np.number)(data)

    mlflow.set_experiment('readmission')
    mlflow.log_param('num_cols', numerical_cols)
    mlflow.log_param('cat_cols', categorical_cols)

    return numerical_cols, categorical_cols

def scalate_var(data:pd.DataFrame)->pd.DataFrame:
    tipificado = StandardScaler().fit(data)  ##Creating a scaler
    x_train = pd.DataFrame(tipificado.transform(data), columns=data.columns)  ##scaling x_train
    return x_train

def drop_hc_cols(data:pd.DataFrame)-> pd.DataFrame:
    """Function to drop higlhy correlated features of breast cancer dataset"""
    cols_to_keep = ['radius_worst', 'concavity_worst', 'fractal_dimension_worst',
                    'texture_worst', 'smoothness_worst', 'symmetry_worst', 'perimeter_se',
                    'smoothness_se', 'area_se', 'texture_se', 'fractal_dimension_se',
                    'symmetry_se']
    data=data[cols_to_keep]
    return data

def imputer_KNN (X_train:pd.DataFrame)->pd.DataFrame:
    """Use knn to impute na values """
    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

    return X_train

def last_processing(data: pd.DataFrame,
                    first: Tuple):
    pipe_transforms = Pipeline(steps= [
        first,
        ('data_scaling', FunctionTransformer(scalate_var)),
        ('outlier_to_na', FunctionTransformer(outlier_tona)),
        ('missing_imputer', FunctionTransformer(imputer_KNN))

    ])
    data_transformed =pd.DataFrame(pipe_transforms.fit_transform(data))
    mlflow.set_experiment('Breast Cancer')
    mlflow.log_param(f"shape train_transformed", data_transformed.shape)

    return pipe_transforms, data_transformed


##function to transform outliers in nas
def outlier_tona(data:pd.DataFrame)->pd.DataFrame:
    """convert outliers to na"""
    for i in data.columns:
        Q1 = data[i].quantile(0.25)
        Q3 = data[i].quantile(0.75)
        IQR = Q3 - Q1
        data[i] = np.where((data[i] < (Q1 - 1.5 * IQR)) | (data[i] > (Q3 + 1.5 * IQR)), np.nan, data[i])

    return data
##function to transform target to category
def to_category(data:pd.DataFrame)->pd.DataFrame:
    """Convert a colum to category"""
    data['diagnosis'].astype('category')
    return data




# function to print shape of the dataframe
def print_shape(data: pd.DataFrame, msg: str = 'Shape =') -> pd.DataFrame:
    """Print shape of dataframe."""
    print(f'{data.shape}{msg}')
    return data

#Function to convert columns of a df to float type
def data_tofloat(data:pd.DataFrame)->pd.DataFrame:
    """Change dataset columns dtype"""
    data=data.astype('float')
    return data
#function to convert columns of a df to stirng type

def data_tostring(data:pd.DataFrame)->pd.DataFrame:
    """Convert dataset columns to type string """
    data = data.astype('string')
    return data

# remove duplicates from data based on a column
def drop_duplicates(data: pd.DataFrame,
                    drop_cols: list) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    return data.drop_duplicates(subset=drop_cols, keep=False)



# function to replace values with np.nan
def replace_missing_values(data: pd.DataFrame,
                           replace_values: list) -> pd.DataFrame:
    """Replace missing values in data with np.nan"""
    data=data.replace(replace_values, np.nan)
    return data

#Function to clean blankspaces after each string of the dataset
def clean_blankspaces(data:pd.DataFrame,cols_to_clean:list)-> pd.DataFrame:
    """Function to delete blankspaces after and before each string of the dataset"""
    for i in cols_to_clean:
        data[i] = data[i].str.strip()
    return data


def drop_exact_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    return data.drop_duplicates(keep=False)

def post_processing(x_in: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """
    General processing to transformed data, like remove duplicates
    important after transformation the data types are numpy ndarray
    Args:
        x_in: x data after transformations
        y_train: y_train
    Returns:
    """
    methods = ["remove duplicates"]
    mlflow.set_experiment('BreastCancer')
    mlflow.log_param('post-processing', methods)

    y = y_train['diagnosis'].to_numpy().reshape(-1, 1)

    data = np.concatenate([x_in, y], axis=1)

    # remove duplicates
    data = np.unique(data, axis=0)
    y_out = data[:, -1]
    x_out = data[:, :-1]
    mlflow.log_param('shape post-processing', x_out.shape)
    return x_out, y_out
