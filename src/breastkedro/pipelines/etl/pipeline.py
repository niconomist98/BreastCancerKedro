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
                name="etl_transforms")
            ])