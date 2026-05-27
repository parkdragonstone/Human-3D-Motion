from __future__ import annotations

import threading
import uuid

from webapp.application.analysis_pipeline_service import AnalysisPipelineService
from webapp.domain.entities import CaptureSession


class AnalysisJobService:
    def __init__(self, analysis_service: AnalysisPipelineService) -> None:
        self._analysis_service = analysis_service
        self._jobs: dict[str, dict] = {}

    def start(self, session: CaptureSession, user_config: dict) -> dict:
        job_id = uuid.uuid4().hex
        self._jobs[job_id] = {"status": "queued", "logs": [], "error": None}
        thread = threading.Thread(
            target=self._run,
            args=(job_id, session, user_config),
            daemon=True,
        )
        thread.start()
        return self.get(job_id) or {"job_id": job_id, **self._jobs[job_id]}

    def get(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return {"job_id": job_id, **job}

    def _run(self, job_id: str, session: CaptureSession, user_config: dict) -> None:
        job = self._jobs[job_id]

        def emit_log(message, level="info"):
            job["logs"].append({"level": level, "message": str(message)})

        try:
            job["status"] = "running"
            self._analysis_service.run(session, user_config, emit_log)
            job["status"] = "completed"
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
            emit_log(str(exc), "error")
