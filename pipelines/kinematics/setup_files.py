from __future__ import annotations

import sys
from pathlib import Path


def get_opensim_setup_dir():
    pipelines_dir = Path(__file__).resolve().parents[1]
    setup_dir = pipelines_dir / "OpenSim_Setup"
    if setup_dir.is_dir():
        return setup_dir

    try:
        pose2sim_mod = sys.modules.get("Pose2Sim", None)
        if pose2sim_mod is None:
            import Pose2Sim as pose2sim_mod  # type: ignore
        pose2sim_path = Path(pose2sim_mod.__file__).resolve().parent
        setup_dir2 = pose2sim_path / "OpenSim_Setup"
        if setup_dir2.is_dir():
            return setup_dir2
    except Exception:
        pass

    raise FileNotFoundError("Cannot locate OpenSim_Setup directory. Expected at: " + str(setup_dir))


def get_model_path(use_simple_model, osim_setup_dir):
    pose_model_file = "Model_Pose2Sim_simple.osim" if use_simple_model else "Model_Pose2Sim_muscles_flex.osim"
    return osim_setup_dir / pose_model_file


def norm_pose_model(pose_model: str) -> str:
    return "".join(pose_model.split("_")).lower()


def find_setup_xml(osim_setup_dir: Path, glob_pat: str, basename_lower: str) -> Path:
    for setup_file in osim_setup_dir.glob(glob_pat):
        if setup_file.name.lower() == basename_lower:
            return setup_file
    raise ValueError(f"OpenSim ?ㅼ젙 XML ??李얠쓣 ???놁뒿?덈떎: {basename_lower}")


def get_markers_path(pose_model, osim_setup_dir):
    pose_model = norm_pose_model(pose_model)
    return find_setup_xml(osim_setup_dir, "Markers_*.xml", f"markers_{pose_model}.xml")


def get_scaling_setup(pose_model, osim_setup_dir):
    pose_model = norm_pose_model(pose_model)
    return find_setup_xml(
        osim_setup_dir,
        "Scaling_Setup_Pose2Sim_*.xml",
        f"scaling_setup_pose2sim_{pose_model}.xml",
    )


def get_ik_setup(pose_model, osim_setup_dir):
    pose_model = norm_pose_model(pose_model)
    name = "ik_setup_pose2sim_withhands_lstm.xml" if pose_model == "lstm" else f"ik_setup_pose2sim_{pose_model}.xml"
    return find_setup_xml(osim_setup_dir, "IK_Setup_Pose2Sim_*.xml", name)
