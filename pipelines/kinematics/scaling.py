from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import opensim
from lxml import etree

from ..utilities import best_coords_for_measurements, euclidean_distance, read_trc, trimmed_mean
from .setup_files import get_markers_path, get_model_path, get_scaling_setup


logger = logging.getLogger(__name__)


def get_kpt_pairs_from_scaling(scaling_root):
    return [
        pair.find("markers").text.strip().split(" ")
        for pair in scaling_root[0].findall(".//MarkerPair")
    ]


def dict_segment_marker_pairs(scaling_root, right_left_symmetry=True):
    segment_markers_dict = {}
    for measurement in scaling_root.findall(".//Measurement"):
        marker_pairs = [pair.find("markers").text.strip().split() for pair in measurement.findall(".//MarkerPair")]
        for body_scale in measurement.findall(".//BodyScale"):
            body_name = body_scale.get("name")
            axes = body_scale.find("axes").text.strip().split()
            for axis in axes:
                body_name_axis = f"{body_name}_{axis}"
                if right_left_symmetry:
                    segment_markers_dict.setdefault(body_name_axis, []).extend(marker_pairs)
                elif body_name.endswith("_r"):
                    marker_pairs_r = [pair for pair in marker_pairs if any([pair[0].upper().startswith("R"), pair[1].upper().startswith("R")])]
                    segment_markers_dict.setdefault(body_name_axis, []).extend(marker_pairs_r)
                elif body_name.endswith("_l"):
                    marker_pairs_l = [pair for pair in marker_pairs if any([pair[0].upper().startswith("L"), pair[1].upper().startswith("L")])]
                    segment_markers_dict.setdefault(body_name_axis, []).extend(marker_pairs_l)
                else:
                    segment_markers_dict.setdefault(body_name_axis, []).extend(marker_pairs)
    return segment_markers_dict


def dict_segment_ratio(
    scaling_root,
    unscaled_model,
    q_coords_scaling,
    markers,
    trimmed_extrema_percent=0.5,
    right_left_symmetry=True,
):
    segment_pairs = get_kpt_pairs_from_scaling(scaling_root)
    trc_segment_lengths = np.array([
        euclidean_distance(
            q_coords_scaling.iloc[:, markers.index(point1) * 3:markers.index(point1) * 3 + 3],
            q_coords_scaling.iloc[:, markers.index(point2) * 3:markers.index(point2) * 3 + 3],
        )
        for point1, point2 in segment_pairs
    ])
    trc_segment_lengths = np.array([
        trimmed_mean(values, trimmed_extrema_percent=trimmed_extrema_percent)
        for values in trc_segment_lengths
    ])

    model_marker_names = [marker.getName() for marker in unscaled_model.getMarkerSet()]
    model_markers = [marker for marker in markers if marker in model_marker_names]
    model_markers_locs = [
        unscaled_model.getMarkerSet().get(marker).getLocationInGround(unscaled_model.getWorkingState()).to_numpy()
        for marker in model_markers
    ]
    model_segment_lengths = np.array([
        euclidean_distance(
            model_markers_locs[model_markers.index(point1)],
            model_markers_locs[model_markers.index(point2)],
        )
        for point1, point2 in segment_pairs
    ])

    segment_ratios = trc_segment_lengths / model_segment_lengths
    segment_markers_dict = dict_segment_marker_pairs(scaling_root, right_left_symmetry=right_left_symmetry)
    segment_ratio_dict_temp = segment_markers_dict.copy()
    segment_ratio_dict_temp.update({
        key: np.mean([segment_ratios[segment_pairs.index(marker_pair)] for marker_pair in segment_markers_dict[key]])
        for key in segment_markers_dict
    })
    segment_ratio_dict = {}
    xyz_keys = list(set(key[:-2] for key in segment_ratio_dict_temp))
    for key in xyz_keys:
        segment_ratio_dict[key] = [
            segment_ratio_dict_temp[key + "_X"],
            segment_ratio_dict_temp[key + "_Y"],
            segment_ratio_dict_temp[key + "_Z"],
        ]
    return segment_ratio_dict


def deactivate_measurements(scaling_root):
    measurement_set = scaling_root.find(".//MeasurementSet/objects")
    for measurement in measurement_set.findall("Measurement"):
        measurement.find("apply").text = "false"


def update_scale_values(scaling_root, segment_ratio_dict):
    scale_set = scaling_root.find(".//ScaleSet/objects")
    for scale in scale_set.findall("Scale"):
        scale_set.remove(scale)
    for segment, scales in segment_ratio_dict.items():
        new_scale = etree.Element("Scale")
        scales_elem = etree.SubElement(new_scale, "scales")
        scales_elem.text = " ".join(map(str, scales))
        segment_elem = etree.SubElement(new_scale, "segment")
        segment_elem.text = segment
        apply_elem = etree.SubElement(new_scale, "apply")
        apply_elem.text = "true"
        scale_set.append(new_scale)


def perform_scaling(
    trc_file,
    pose_model,
    kinematics_dir,
    osim_setup_dir,
    use_simple_model=False,
    right_left_symmetry=True,
    subject_height=1.75,
    subject_mass=70,
    remove_scaling_setup=True,
    fastest_frames_to_remove_percent=0.1,
    close_to_zero_speed_m=0.2,
    large_hip_knee_angles=45,
    trimmed_extrema_percent=0.5,
):
    try:
        opensim.ModelVisualizer.addDirToGeometrySearchPaths(str(osim_setup_dir / "Geometry"))
        unscaled_model_path = get_model_path(use_simple_model, osim_setup_dir)
        if not unscaled_model_path:
            raise ValueError(f"Unscaled OpenSim model not found at: {unscaled_model_path}")
        unscaled_model = opensim.Model(str(unscaled_model_path))
        markerset = opensim.MarkerSet(str(get_markers_path(pose_model, osim_setup_dir)))
        unscaled_model.set_MarkerSet(markerset)
        unscaled_model.initSystem()
        scaled_model_path = str((kinematics_dir / (trc_file.stem + ".osim")).resolve())
        unscaled_model.printToXML(scaled_model_path)

        scaling_path = get_scaling_setup(pose_model, osim_setup_dir)
        scaling_tree = etree.parse(scaling_path)
        scaling_root = scaling_tree.getroot()
        scaling_path_temp = str(kinematics_dir / (trc_file.stem + "_scaling_setup.xml"))

        q_coords, _, _, markers, _ = read_trc(trc_file)
        q_coords_filtered = best_coords_for_measurements(
            q_coords,
            markers,
            fastest_frames_to_remove_percent=fastest_frames_to_remove_percent,
            large_hip_knee_angles=large_hip_knee_angles,
            close_to_zero_speed=close_to_zero_speed_m,
        )
        if q_coords_filtered.size == 0:
            logger.warning(f"\nNo frames left after removing fastest frames, frames with null speed, and frames with large hip and knee angles for {trc_file}. The person may be static, or crouched, or incorrectly detected.")
            logger.warning("Running with fastest_frames_to_remove_percent=0, close_to_zero_speed_m=0, large_hip_knee_angles=0, trimmed_extrema_percent=0. You can edit these parameters in your Config.toml file.\n")
            q_coords_filtered = q_coords

        segment_ratio_dict = dict_segment_ratio(
            scaling_root,
            unscaled_model,
            q_coords_filtered,
            markers,
            trimmed_extrema_percent=trimmed_extrema_percent,
            right_left_symmetry=right_left_symmetry,
        )
        scaling_root[0].find("mass").text = str(subject_mass)
        scaling_root[0].find("height").text = str(subject_height)
        scaling_root[0].find("GenericModelMaker").find("model_file").text = scaled_model_path
        scaling_root[0].find(".//scaling_order").text = " manualScale measurements"
        deactivate_measurements(scaling_root)
        update_scale_values(scaling_root, segment_ratio_dict)
        for marker_file in scaling_root[0].findall(".//marker_file"):
            marker_file.text = "Unassigned"
        scaling_root[0].find("ModelScaler").find("output_model_file").text = scaled_model_path

        etree.indent(scaling_tree, space="\t", level=0)
        scaling_tree.write(scaling_path_temp, pretty_print=True, xml_declaration=True, encoding="utf-8")
        opensim.ScaleTool(scaling_path_temp).run()
        if remove_scaling_setup:
            Path(scaling_path_temp).unlink()
    except Exception as exc:
        logger.error(f"Error during scaling for {trc_file}: {exc}.")
        raise
