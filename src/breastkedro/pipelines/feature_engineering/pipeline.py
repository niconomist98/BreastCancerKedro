"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=pre_processing,
                inputs=["x_train","y_train", "parameters"],
                outputs=["x_train_out", "y_train_out"],
                name="pre_processing"),

            node(
                func=first_processing,
                inputs=["x_train_out","parameters"],
                outputs=["data_first", "first_processing_pipeline"],
                name = "first_processing"),

            node(
                func=last_processing,
                inputs=["x_train_out",
                        "first_processing_pipeline"],
                outputs=["column_transformers_pipeline", "x_train_transformed"],
                name="cols_transforms_pipeline"),

            node(
                func=post_processing,
                inputs=["x_train_transformed",
                        "y_train_out"],
                outputs=["x_train_model_input",
                         "y_train_model_input"],
                name="post_processing"
            )])



