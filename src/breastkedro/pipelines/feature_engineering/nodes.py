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
    
    pipe_functions = [
                ('Drop unnecesary columns',FunctionTransformer(drop_cols,
                                                    kw_args={'cols_to_keep':cols_to_keep}
                                                ))

    ]
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
def drop_cols(data: pd.DataFrame,
                       cols_to_keep: list) -> pd.DataFrame:
    """Filter columns from data."""
    data = data[cols_to_keep]
    return data