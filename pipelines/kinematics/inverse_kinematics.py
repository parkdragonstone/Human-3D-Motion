from __future__ import annotations

import logging
from pathlib import Path

import opensim
from lxml import etree

from ..utilities import read_trc
from .setup_files import get_ik_setup


logger = logging.getLogger(__name__)


def perform_ik(trc_file, kinematics_dir, osim_setup_dir, pose_model, remove_IK_setup=True):
    try:
        ik_path = get_ik_setup(pose_model, osim_setup_dir)
        ik_path_temp = str(kinematics_dir / (trc_file.stem + "_ik_setup.xml"))
        scaled_model_path = (kinematics_dir / (trc_file.stem + ".osim")).resolve()
        output_motion_file = Path(kinematics_dir, trc_file.stem + ".mot").resolve()
        if not trc_file.exists():
            raise FileNotFoundError(f"TRC file does not exist: {trc_file}")
        _, _, time_col, _, _ = read_trc(trc_file)
        start_time, end_time = time_col.iloc[0], time_col.iloc[-1]

        ik_tree = etree.parse(ik_path)
        ik_root = ik_tree.getroot()
        ik_root.find(".//model_file").text = str(scaled_model_path)
        ik_root.find(".//time_range").text = f"{start_time} {end_time}"
        ik_root.find(".//output_motion_file").text = str(output_motion_file)
        ik_root.find(".//marker_file").text = str(trc_file.resolve())
        ik_tree.write(ik_path_temp)

        opensim.InverseKinematicsTool(str(ik_path_temp)).run()
        if remove_IK_setup:
            Path(ik_path_temp).unlink()
    except Exception as exc:
        logger.error(f"Error during IK for {trc_file}: {exc}")
        raise
