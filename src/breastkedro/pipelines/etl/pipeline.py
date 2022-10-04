"""
This is a boilerplate pipeline 'etl'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import get_data
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= get_data,
                inputs=["data"],
                outputs="data_raw",
                name="extracción",
            )])
