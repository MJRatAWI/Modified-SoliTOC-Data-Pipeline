"""Kedro pipeline for SoliTOC zone analysis."""

from kedro.pipeline import Pipeline, node, pipeline

from my_pipeline.nodes.nodes import run_zone_analysis


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_zone_analysis,
                inputs="params:analysis",
                outputs=None,
                name="run_zone_analysis",
            )
        ]
    )
