from __future__ import annotations


class PipelineAnalysisConfigProvider:
    def default_config(self) -> dict:
        from pipelines.config import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    def merge_config(self, user_config: dict) -> dict:
        from pipelines.config import DEFAULT_CONFIG, deep_merge

        return deep_merge(DEFAULT_CONFIG, user_config if isinstance(user_config, dict) else {})
