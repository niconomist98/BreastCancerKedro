"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [   node(
                func=pre_processing,
                inputs=["x_train","y_train", "parameters"],
                outputs=["x_train_out", "y_train_out"],
                name="pre_processing",
            )

        ]
    )