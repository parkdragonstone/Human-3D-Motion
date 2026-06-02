"""OpenSim ?ㅼ??쇰쭅쨌??린援ы븰 (TRC 湲곕컲)."""
import locale
import logging
import time
from pathlib import Path

import opensim

from ..kinematics_csv import export_combined_kinematics_csv, resolve_keypoint_trc
from .inverse_kinematics import perform_ik as perform_IK
from .scaling import (
    deactivate_measurements,
    dict_segment_marker_pairs,
    dict_segment_ratio,
    get_kpt_pairs_from_scaling,
    perform_scaling,
    update_scale_values,
)
from .setup_files import (
    find_setup_xml as _find_setup_xml,
    get_ik_setup as get_IK_Setup,
    get_markers_path,
    get_model_path,
    get_opensim_setup_dir,
    get_scaling_setup,
    norm_pose_model as _norm_pose_model,
)
from ..utilities import natural_sort_key


locale.setlocale(locale.LC_NUMERIC, "C")
logger = logging.getLogger(__name__)


def run_kinematics(config_dict, emit_log=None):
    """TRC 湲곗? OpenSim ?ㅼ??쇰쭅 諛?IK ?ㅽ뻾."""

    def _log(text, level="info"):
        if callable(emit_log):
            emit_log(text, level)
        else:
            logger.info(text)

    def _float_cfg(section, *keys, default=None):
        settings = section or {}
        for key in keys:
            if key not in settings:
                continue
            value = settings[key]
            if value is None or (isinstance(value, str) and not str(value).strip()):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        if default is not None:
            return float(default)
        raise ValueError(f"?꾩닔 ?ㅼ젙 ?꾨씫: {keys}")

    project_dir = config_dict.get("paths").get("project_dir")
    kin = config_dict.get("kinematics") or {}
    subject = config_dict.get("subject") or {}
    base = config_dict.get("base") or {}

    use_augmentation = kin.get("use_augmentation")
    use_simple_model = kin.get("use_simple_model")
    right_left_symmetry = kin.get("right_left_symmetry")
    subject_height = _float_cfg(subject, "height", default=170) / 100
    subject_mass = _float_cfg(subject, "weight", default=70)
    fastest_frames_to_remove_percent = _float_cfg(
        kin,
        "fastest_frames_to_remove_percent",
        "fast_frames_to_remove_percent",
        default=0.1,
    )
    close_to_zero_speed = _float_cfg(kin, "close_to_zero_speed_m", default=0.2)
    large_hip_knee_angles = _float_cfg(kin, "large_hip_knee_angles", default=45.0)
    trimmed_extrema_percent = _float_cfg(kin, "trimmed_extrema_percent", default=0.5)
    kinematics_filter = kin.get("filter") or {}
    remove_scaling_setup = kin.get("remove_individual_scaling_setup")
    remove_ik_setup = kin.get("remove_individual_ik_setup")

    pose3d_dir = Path(project_dir) / "pose-3d"
    kinematics_dir = Path(project_dir) / "kinematics"
    kinematics_dir.mkdir(parents=True, exist_ok=True)
    osim_setup_dir = get_opensim_setup_dir()

    opensim_logs_file = kinematics_dir / "opensim_logs.txt"
    opensim.Logger.setLevelString("Info")
    opensim.Logger.removeFileSink()
    opensim.Logger.addFileSink(str(opensim_logs_file))

    trc_files = []
    if use_augmentation:
        trc_files = [path for path in pose3d_dir.glob("*.trc") if "_LSTM" in path.name]
        if len(trc_files) == 0:
            use_augmentation = False
            logger.warning("No LSTM trc files found. Using non augmented trc files instead.")
    if len(trc_files) == 0:
        trc_files = [
            path for path in pose3d_dir.glob("*.trc")
            if "_LSTM" not in path.name and "_filt" in path.name and "_scaling" not in path.name
        ]
    if len(trc_files) == 0:
        trc_files = [
            path for path in pose3d_dir.glob("*.trc")
            if "_LSTM" not in path.name and "_scaling" not in path.name
        ]
    if len(trc_files) == 0:
        raise ValueError(f"No trc files found in {pose3d_dir}.")
    trc_files = sorted(trc_files, key=natural_sort_key)

    pose_model = "LSTM" if use_augmentation else "HALPE_26"
    subject_height = [subject_height] if not isinstance(subject_height, list) else subject_height
    subject_mass = [subject_mass] if not isinstance(subject_mass, list) else subject_mass

    for subject_index, trc_file in enumerate(trc_files):
        _log(f"Processing TRC file: {trc_file.resolve()}")

        _log("\nScaling...")
        perform_scaling(
            trc_file,
            pose_model,
            kinematics_dir,
            osim_setup_dir,
            use_simple_model,
            right_left_symmetry=right_left_symmetry,
            subject_height=subject_height[subject_index],
            subject_mass=subject_mass[subject_index],
            remove_scaling_setup=remove_scaling_setup,
            fastest_frames_to_remove_percent=fastest_frames_to_remove_percent,
            large_hip_knee_angles=large_hip_knee_angles,
            trimmed_extrema_percent=trimmed_extrema_percent,
            close_to_zero_speed_m=close_to_zero_speed,
        )
        _log(f"\tDone. OpenSim logs saved to {opensim_logs_file.resolve()}.")
        _log(f"\tScaled model saved to {(kinematics_dir / (trc_file.stem + '_scaled.osim')).resolve()}")

        _log("\nInverse Kinematics...")
        start_time = time.time()
        perform_IK(trc_file, kinematics_dir, osim_setup_dir, pose_model, remove_IK_setup=remove_ik_setup)
        end_time = time.time()
        _log(f"\tIK took {round(end_time - start_time, 2)} seconds for {trc_file.name}.")
        _log(f"\tDone. OpenSim logs saved to {opensim_logs_file.resolve()}.")
        mot_path = kinematics_dir / (trc_file.stem + ".mot")
        _log(f"\tJoint angle data saved to {mot_path.resolve()}")
        keypoint_trc_path = resolve_keypoint_trc(Path(project_dir), trc_file)
        combined_csv_path = export_combined_kinematics_csv(
            Path(project_dir),
            mot_path,
            keypoint_trc_path,
            kinematics_filter,
            subject_metadata=subject,
            fps=base.get("fps"),
            motion=base.get("motion", "Baseball-Pitching"),
            walking_direction=base.get("walking_direction", "-z"),
        )
        _log(f"\tCombined keypoint and kinematics CSV saved to {combined_csv_path.resolve()}\n")
