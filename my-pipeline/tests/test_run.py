from pathlib import Path
from kedro.framework.startup import bootstrap_project

from my_pipeline.pipeline_registry import register_pipelines

class TestKedroRun:
    def test_default_pipeline_is_registered(self):
        bootstrap_project(Path.cwd())
        pipelines = register_pipelines()
        assert "__default__" in pipelines
        assert len(pipelines["__default__"].nodes) == 1

