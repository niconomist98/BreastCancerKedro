"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            #node(
            #    func=get_data,
            #    inputs="parameters",
            #    outputs="Data_raw",
            #    name="get_data_raw",
            #),
            node(
                func=etl_processing,
                inputs=["data_original", "parameters"],
                outputs="data_preprocessed",
                name="etl_transforms"),
            node (
                func=data_integrity_validation,
                inputs=["data_preprocessed","parameters"],
                outputs="data_integrity_check",
                name="data_integrity_validation"),

            node(
                func=split_data,
                inputs=["data_integrity_check", "parameters"],
                outputs=["x_train_split",
                         "x_test_split",
                         "y_train_split",
                         "y_test_split"],
                name="split-train_test",
            )

                                              ])
