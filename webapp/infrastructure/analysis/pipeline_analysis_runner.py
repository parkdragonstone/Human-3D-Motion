from __future__ import annotations

from webapp.domain.ports import LogEmitter


class PipelineAnalysisRunner:
    def run(self, config: dict, emit_log: LogEmitter) -> None:
        from pipelines.filtering import run_filtering
        from pipelines.kinematics import run_kinematics
        from pipelines.markerAugmentation import run_markerAugmentation
        from pipelines.poseEstimation import run_poseEstimation
        from pipelines.reconstruction import run_3d_lifting

        emit_log("Pose estimation started", "info")
        run_poseEstimation(config, emit_log=emit_log)
        emit_log("Reconstruction started", "info")
        run_3d_lifting(config, emit_log=emit_log)
        emit_log("Filtering started", "info")
        run_filtering(config, emit_log=emit_log)
        emit_log("Marker augmentation started", "info")
        run_markerAugmentation(config, emit_log=emit_log)
        emit_log("Kinematics started", "info")
        run_kinematics(config, emit_log=emit_log)
        emit_log("Analysis complete", "info")
