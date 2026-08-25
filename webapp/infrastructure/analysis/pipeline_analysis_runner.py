from __future__ import annotations

from webapp.domain.ports import LogEmitter


class PipelineAnalysisRunner:
    def run(self, config: dict, emit_log: LogEmitter) -> None:
        from pipelines.calibration.auto_calibration import bundle_from_payload, run_auto_calibration
        from pipelines.filtering import run_filtering
        from pipelines.kinematics import run_kinematics
        from pipelines.markerAugmentation import run_markerAugmentation
        from pipelines.poseEstimation import run_poseEstimation
        from pipelines.reconstruction import has_full_calibration, run_3d_lifting

        emit_log("Pose estimation started", "info")
        run_poseEstimation(config, emit_log=emit_log)

        # Reconstruction needs camera geometry. Without a calibration file the person
        # just detected in the footage becomes the calibration target, so this has to
        # happen after pose estimation and before lifting.
        if not has_full_calibration(config.get("calibration")):
            emit_log("Auto calibration started", "info")
            payload = run_auto_calibration(config, emit_log=emit_log)
            if payload is None:
                raise RuntimeError(
                    "auto_calibration_failed: no calibration file was supplied and the "
                    "cameras could not be calibrated from the recorded motion"
                )
            config["calibration"] = bundle_from_payload(payload)

        emit_log("Reconstruction started", "info")
        run_3d_lifting(config, emit_log=emit_log)
        emit_log("Filtering started", "info")
        run_filtering(config, emit_log=emit_log)
        emit_log("Marker augmentation started", "info")
        run_markerAugmentation(config, emit_log=emit_log)
        emit_log("Kinematics started", "info")
        run_kinematics(config, emit_log=emit_log)
        emit_log("Analysis complete", "info")
