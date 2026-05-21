"""OpenSim 스케일링·역기구학 (TRC 기반)."""
import sys
import time
from pathlib import Path
import logging

import numpy as np
import opensim
from lxml import etree

np.set_printoptions(legacy='1.21')

from .utilities import (
    natural_sort_key,
    euclidean_distance,
    trimmed_mean,
    read_trc,
    best_coords_for_measurements,
)
from .kinematics_csv import export_combined_kinematics_csv, resolve_keypoint_trc

import locale 
locale.setlocale(locale.LC_NUMERIC, 'C')

logger = logging.getLogger(__name__)


def get_opensim_setup_dir():
    '''
    Locate the OpenSim setup directory.

    OUTPUTS:
    - Path: The path to the OpenSim setup directory.
    '''

    # 1) Local (this repo) path: <repo>/PoseAnalysis/OpenSim_Setup
    poseanalysis_dir = Path(__file__).resolve().parent
    setup_dir = poseanalysis_dir / 'OpenSim_Setup'
    if setup_dir.is_dir():
        return setup_dir

    # 2) Fallback: try Pose2Sim package if installed
    try:
        pose2sim_mod = sys.modules.get('Pose2Sim', None)
        if pose2sim_mod is None:
            import Pose2Sim as pose2sim_mod  # type: ignore
        pose2sim_path = Path(pose2sim_mod.__file__).resolve().parent
        setup_dir2 = pose2sim_path / 'OpenSim_Setup'
        if setup_dir2.is_dir():
            return setup_dir2
    except Exception:
        pass

    raise FileNotFoundError(
        'Cannot locate OpenSim_Setup directory. Expected at: ' + str(setup_dir)
    )


def get_model_path(use_simple_model, osim_setup_dir):
    '''
    Retrieve the path of the OpenSim model file.

    INPUTS:
    - pose_model (str): Name of the model
    - osim_setup_dir (Path): Path to the OpenSim setup directory.

    OUTPUTS:
    - pose_model_path: (Path) Path to the OpenSim model file.
    '''

    if use_simple_model:
        pose_model_file = 'Model_Pose2Sim_simple.osim'
    else:
        pose_model_file = 'Model_Pose2Sim_muscles_flex.osim'

    unscaled_model_path = osim_setup_dir / pose_model_file

    return unscaled_model_path


def _norm_pose_model(pose_model: str) -> str:
    return ''.join(pose_model.split('_')).lower()


def _find_setup_xml(osim_setup_dir: Path, glob_pat: str, basename_lower: str) -> Path:
    for f in osim_setup_dir.glob(glob_pat):
        if f.name.lower() == basename_lower:
            return f
    raise ValueError(f"OpenSim 설정 XML 을 찾을 수 없습니다: {basename_lower}")


def get_markers_path(pose_model, osim_setup_dir):
    pm = _norm_pose_model(pose_model)
    return _find_setup_xml(osim_setup_dir, 'Markers_*.xml', f'markers_{pm}.xml')


def get_scaling_setup(pose_model, osim_setup_dir):
    pm = _norm_pose_model(pose_model)
    return _find_setup_xml(
        osim_setup_dir, 'Scaling_Setup_Pose2Sim_*.xml',
        f'scaling_setup_pose2sim_{pm}.xml',
    )


def get_IK_Setup(pose_model, osim_setup_dir):
    pm = _norm_pose_model(pose_model)
    if pm == 'lstm':
        name = 'ik_setup_pose2sim_withhands_lstm.xml'
    else:
        name = f'ik_setup_pose2sim_{pm}.xml'
    return _find_setup_xml(osim_setup_dir, 'IK_Setup_Pose2Sim_*.xml', name)


def get_kpt_pairs_from_scaling(scaling_root):
    '''
    Get all marker pairs from the scaling setup file.

    INPUTS:
    - scaling_root (Element): The root element of the scaling setup file.

    OUTPUTS:
    - pairs: A list of marker pairs.
    '''

    pairs = [pair.find('markers').text.strip().split(' ') 
             for pair in scaling_root[0].findall(".//MarkerPair")]

    return pairs


def dict_segment_marker_pairs(scaling_root, right_left_symmetry=True):
    '''
    Get a dictionary of segment names and their corresponding marker pairs.

    INPUTS:
    - scaling_root (Element): The root element of the scaling setup file.
    - right_left_symmetry (bool): Whether to consider right and left side of equal size.

    OUTPUTS:
    - segment_markers_dict: A dictionary of segment names and their corresponding marker pairs.
    '''

    segment_markers_dict = {}
    for measurement in scaling_root.findall(".//Measurement"):
        # Collect all marker pairs for this measurement
        marker_pairs = [pair.find('markers').text.strip().split() for pair in measurement.findall(".//MarkerPair")]

        # Collect all body scales for this measurement
        for body_scale in measurement.findall(".//BodyScale"):
            body_name = body_scale.get('name')
            axes = body_scale.find('axes').text.strip().split()
            for axis in axes:
                body_name_axis = f"{body_name}_{axis}"
                if right_left_symmetry:
                    segment_markers_dict.setdefault(body_name_axis, []).extend(marker_pairs)
                else:
                    if body_name.endswith('_r'):
                        marker_pairs_r = [pair for pair in marker_pairs if any([pair[0].upper().startswith('R'), pair[1].upper().startswith('R')])]
                        segment_markers_dict.setdefault(body_name_axis, []).extend(marker_pairs_r)
                    elif body_name.endswith('_l'):
                        marker_pairs_l = [pair for pair in marker_pairs if any([pair[0].upper().startswith('L'), pair[1].upper().startswith('L')])]
                        segment_markers_dict.setdefault(body_name_axis, []).extend(marker_pairs_l)
                    else:
                        segment_markers_dict.setdefault(body_name_axis, []).extend(marker_pairs)

    return segment_markers_dict


def dict_segment_ratio(scaling_root, unscaled_model, Q_coords_scaling, markers, trimmed_extrema_percent=0.5, right_left_symmetry=True):
    '''
    Calculate the ratios between the size of the actual segment and the size of the model segment.
    X, Y, and Z ratios are calculated separately if the original scaling setup file asks for it.

    INPUTS:
    - scaling_root (Element): The root element of the scaling setup file.
    - unscaled_model (Model): The original OpenSim model before scaling.
    - Q_coords_scaling (DataFrame): The triangulated coordinates of the markers.
    - markers (list): The list of marker names.
    - trimmed_extrema_percent (float): The proportion of the most extreme segment values to remove before calculating their mean.
    - right_left_symmetry (bool): Whether to consider right and left side of equal size.

    OUTPUTS:
    - segment_ratio_dict: A dictionary of segment names and their corresponding X, Y, and Z ratios.
    '''

    segment_pairs = get_kpt_pairs_from_scaling(scaling_root)

    # Get median segment lengths from Q_coords_scaling. Trimmed mean works better than mean or median
    trc_segment_lengths = np.array([euclidean_distance(Q_coords_scaling.iloc[:,markers.index(pt1)*3:markers.index(pt1)*3+3], 
                        Q_coords_scaling.iloc[:,markers.index(pt2)*3:markers.index(pt2)*3+3]) 
                        for (pt1,pt2) in segment_pairs])
    trc_segment_lengths = np.array([trimmed_mean(arr, trimmed_extrema_percent=trimmed_extrema_percent) for arr in trc_segment_lengths])

    # Get model segment lengths
    model_markers = [marker for marker in markers if marker in [m.getName() for m in unscaled_model.getMarkerSet()]]
    model_markers_locs = [unscaled_model.getMarkerSet().get(marker).getLocationInGround(unscaled_model.getWorkingState()).to_numpy() for marker in model_markers]
    model_segment_lengths = np.array([euclidean_distance(model_markers_locs[model_markers.index(pt1)], 
                                                model_markers_locs[model_markers.index(pt2)]) 
                                                for (pt1,pt2) in segment_pairs])
    
    # Calculate ratio for each segment
    segment_ratios = trc_segment_lengths / model_segment_lengths
    segment_markers_dict = dict_segment_marker_pairs(scaling_root, right_left_symmetry=right_left_symmetry)
    segment_ratio_dict_temp = segment_markers_dict.copy()
    segment_ratio_dict_temp.update({key: np.mean([segment_ratios[segment_pairs.index(k)] 
                                            for k in segment_markers_dict[key]]) 
                                for key in segment_markers_dict.keys()})
    # Merge X, Y, Z ratios into single key
    segment_ratio_dict={}
    xyz_keys = list(set([key[:-2] for key in segment_ratio_dict_temp.keys()]))
    for key in xyz_keys:
        segment_ratio_dict[key] = [segment_ratio_dict_temp[key+'_X'], segment_ratio_dict_temp[key+'_Y'], segment_ratio_dict_temp[key+'_Z']]
    
    return segment_ratio_dict


def deactivate_measurements(scaling_root):
    '''
    Deactivate all scalings based on marker positions (called 'measurements' in OpenSim) in the scaling setup file.
    (will use scaling based on segment sizes instead (called 'manual' in OpenSim))

    INPUTS:
    - scaling_root (Element): The root element of the scaling setup file.

    OUTPUTS:
    - scaling_root with deactivated measurements.
    '''
    
    measurement_set = scaling_root.find(".//MeasurementSet/objects")
    for measurement in measurement_set.findall('Measurement'):
            apply_elem = measurement.find('apply')
            apply_elem.text = 'false'


def update_scale_values(scaling_root, segment_ratio_dict):
    '''
    Remove previous scaling values ('manual') and 
    add new scaling values based on calculated segment ratios.

    INPUTS:
    - scaling_root (Element): The root element of the scaling setup file.
    - segment_ratio_dict (dict): A dictionary of segment names and their corresponding X, Y, and Z ratios.

    OUTPUTS:
    - scaling_root with updated scaling values.
    '''
    
    # Get the ScaleSet/objects element
    scale_set = scaling_root.find(".//ScaleSet/objects")

    # Remove all existing Scale elements
    for scale in scale_set.findall('Scale'):
        scale_set.remove(scale)

    # Add new Scale elements based on scale_dict
    for segment, scales in segment_ratio_dict.items():
        new_scale = etree.Element('Scale')
        # scales
        scales_elem = etree.SubElement(new_scale, 'scales')
        scales_elem.text = ' '.join(map(str, scales))
        # segment name
        segment_elem = etree.SubElement(new_scale, 'segment')
        segment_elem.text = segment
        # apply True
        apply_elem = etree.SubElement(new_scale, 'apply')
        apply_elem.text = 'true'

        scale_set.append(new_scale)
        

def perform_scaling(trc_file, pose_model, kinematics_dir, osim_setup_dir, 
                    use_simple_model=False, right_left_symmetry=True, subject_height=1.75, subject_mass=70, 
                    remove_scaling_setup=True, fastest_frames_to_remove_percent=0.1,close_to_zero_speed_m=0.2, large_hip_knee_angles=45, trimmed_extrema_percent=0.5):
    '''
    Perform model scaling based on the (not necessarily static) TRC file:
    - Remove 10% fastest frames (potential outliers)
    - Remove frames where coordinate speed is null (person probably out of frame)
    - Remove 40% most extreme calculated segment values (potential outliers)
    - For each segment, scale on the mean of the remaining segment values
    
    INPUTS:
    - trc_file (Path): The path to the TRC file.
    - kinematics_dir (Path): The directory where the kinematics files are saved.
    - osim_setup_dir (Path): The directory where the OpenSim setup and model files are stored.
    - pose_model (str): The name of the model.
    - use_simple_model (bool): Whether to use the model without constraints and muscles.
    - right_left_symmetry (bool): Whether to consider right and left side of equal size.
    - subject_height (float): The height of the subject.
    - subject_mass (float): The mass of the subject.
    - remove_scaling_setup (bool): Whether to remove the scaling setup file after scaling.
    - fastest_frames_to_remove_percent (float): Fasters frames may be outliers
    - large_hip_knee_angles (float): Imprecise coordinates when person is crouching
    - trimmed_extrema_percent (float): Proportion of the most extreme segment values to remove before calculating their mean
    
    OUTPUTS:
    - A scaled OpenSim model file.
    '''

    try:
        # Load model
        opensim.ModelVisualizer.addDirToGeometrySearchPaths(str(osim_setup_dir / 'Geometry'))
        unscaled_model_path = get_model_path(use_simple_model, osim_setup_dir)
        if not unscaled_model_path:
            raise ValueError(f"Unscaled OpenSim model not found at: {unscaled_model_path}")
        unscaled_model = opensim.Model(str(unscaled_model_path))
        # Add markers to model
        markers_path = get_markers_path(pose_model, osim_setup_dir)
        markerset = opensim.MarkerSet(str(markers_path))
        unscaled_model.set_MarkerSet(markerset)
        # Initialize and save model with markers
        unscaled_model.initSystem()
        scaled_model_path = str((kinematics_dir / (trc_file.stem + '.osim')).resolve())
        unscaled_model.printToXML(scaled_model_path)

        # Load scaling setup
        scaling_path = get_scaling_setup(pose_model, osim_setup_dir)
        scaling_tree = etree.parse(scaling_path)
        scaling_root = scaling_tree.getroot()
        scaling_path_temp = str(kinematics_dir / (trc_file.stem + '_scaling_setup.xml'))
        
        # Remove fastest frames, frames with null speed, and frames with large hip and knee angles
        Q_coords, _, _, markers, _ = read_trc(trc_file)
        Q_coords_low_speeds_low_angles = best_coords_for_measurements(Q_coords, markers, fastest_frames_to_remove_percent=fastest_frames_to_remove_percent, large_hip_knee_angles=large_hip_knee_angles, close_to_zero_speed=close_to_zero_speed_m)

        if Q_coords_low_speeds_low_angles.size == 0:
            logger.warning(f"\nNo frames left after removing fastest frames, frames with null speed, and frames with large hip and knee angles for {trc_file}. The person may be static, or crouched, or incorrectly detected.")
            logger.warning("Running with fastest_frames_to_remove_percent=0, close_to_zero_speed_m=0, large_hip_knee_angles=0, trimmed_extrema_percent=0. You can edit these parameters in your Config.toml file.\n")
            Q_coords_low_speeds_low_angles = Q_coords

        # Get manual scale values (mean from remaining frames after trimming the 20% most extreme values)
        segment_ratio_dict = dict_segment_ratio(scaling_root, unscaled_model, Q_coords_low_speeds_low_angles, markers, 
                                                trimmed_extrema_percent=trimmed_extrema_percent, right_left_symmetry=right_left_symmetry)

        # Update scaling setup file
        scaling_root[0].find('mass').text = str(subject_mass)
        scaling_root[0].find('height').text = str(subject_height)
        scaling_root[0].find('GenericModelMaker').find('model_file').text = scaled_model_path
        scaling_root[0].find(".//scaling_order").text = ' manualScale measurements'
        deactivate_measurements(scaling_root)
        update_scale_values(scaling_root, segment_ratio_dict)
        for mk_f in scaling_root[0].findall(".//marker_file"): mk_f.text = "Unassigned"
        scaling_root[0].find('ModelScaler').find('output_model_file').text = scaled_model_path

        etree.indent(scaling_tree, space='\t', level=0)
        scaling_tree.write(scaling_path_temp, pretty_print=True, xml_declaration=True, encoding='utf-8')

        # Run scaling
        opensim.ScaleTool(scaling_path_temp).run()

        # Remove scaling setup
        if remove_scaling_setup:
            Path(scaling_path_temp).unlink()

    except Exception as e:
        logger.error(f"Error during scaling for {trc_file}: {e}.")
        raise


def perform_IK(trc_file, kinematics_dir, osim_setup_dir, pose_model, remove_IK_setup=True):
    '''
    Perform inverse kinematics based on a TRC file and a scaled OpenSim model:
    - Model markers follow the triangulated markers while respecting the model kinematic constraints
    - Joint angles are computed

    INPUTS:
    - trc_file (Path): The path to the TRC file.
    - kinematics_dir (Path): The directory where the kinematics files are saved.
    - osim_setup_dir (Path): The directory where the OpenSim setup and model files are stored.
    - pose_model (str): The name of the model.
    - remove_IK_setup (bool): Whether to remove the IK setup file after running IK.

    OUTPUTS:
    - A joint angle data file (.mot).
    '''

    try:
        # Retrieve data
        ik_path = get_IK_Setup(pose_model, osim_setup_dir)
        ik_path_temp =  str(kinematics_dir / (trc_file.stem + '_ik_setup.xml'))
        scaled_model_path = (kinematics_dir / (trc_file.stem + '.osim')).resolve()
        output_motion_file = Path(kinematics_dir, trc_file.stem + '.mot').resolve()
        if not trc_file.exists():
            raise FileNotFoundError(f"TRC file does not exist: {trc_file}")
        _, _, time_col, _, _ = read_trc(trc_file)
        start_time, end_time = time_col.iloc[0], time_col.iloc[-1]

        # Update IK setup file
        ik_tree = etree.parse(ik_path)
        ik_root = ik_tree.getroot()
        ik_root.find('.//model_file').text = str(scaled_model_path)
        ik_root.find('.//time_range').text = f'{start_time} {end_time}'
        ik_root.find('.//output_motion_file').text = str(output_motion_file)
        ik_root.find('.//marker_file').text = str(trc_file.resolve())
        ik_tree.write(ik_path_temp)

        # Run IK
        opensim.InverseKinematicsTool(str(ik_path_temp)).run()

        # Remove IK setup
        if remove_IK_setup:
            Path(ik_path_temp).unlink()

    except Exception as e:
        logger.error(f"Error during IK for {trc_file}: {e}")
        raise


def run_kinematics(config_dict, emit_log=None):
    """TRC 기준 OpenSim 스케일링 및 IK 실행."""

    def _log(text, level='info'):
        if callable(emit_log):
            emit_log(text, level)
        else:
            logger.info(text)

    def _float_cfg(section, *keys, default=None):
        d = section or {}
        for k in keys:
            if k not in d:
                continue
            v = d[k]
            if v is None or (isinstance(v, str) and not str(v).strip()):
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        if default is not None:
            return float(default)
        raise ValueError(f'필수 설정 누락: {keys}')

    project_dir = config_dict.get('paths').get('project_dir')
    kin = config_dict.get('kinematics') or {}
    sub = config_dict.get('subject') or {}
    base = config_dict.get('base') or {}

    use_augmentation = kin.get('use_augmentation')
    use_simple_model = kin.get('use_simple_model')
    right_left_symmetry = kin.get('right_left_symmetry')
    subject_height = _float_cfg(sub, 'height', default=170) / 100
    subject_mass = _float_cfg(sub, 'weight', default=70)
    fastest_frames_to_remove_percent = _float_cfg(
        kin, 'fastest_frames_to_remove_percent', 'fast_frames_to_remove_percent', default=0.1,
    )
    close_to_zero_speed = _float_cfg(kin, 'close_to_zero_speed_m', default=0.2)
    large_hip_knee_angles = _float_cfg(kin, 'large_hip_knee_angles', default=45.0)
    trimmed_extrema_percent = _float_cfg(kin, 'trimmed_extrema_percent', default=0.5)
    kinematics_filter = kin.get('filter') or {}
    remove_scaling_setup = kin.get('remove_individual_scaling_setup')
    remove_IK_setup = kin.get('remove_individual_ik_setup')

    pose3d_dir = Path(project_dir) / 'pose-3d'
    kinematics_dir = Path(project_dir) / 'kinematics'
    kinematics_dir.mkdir(parents=True, exist_ok=True)
    osim_setup_dir = get_opensim_setup_dir()
    
    # OpenSim logs saved to a different file
    opensim_logs_file = kinematics_dir / 'opensim_logs.txt'
    opensim.Logger.setLevelString('Info')
    opensim.Logger.removeFileSink()
    opensim.Logger.addFileSink(str(opensim_logs_file))

    # Find all trc files
    trc_files = []
    if use_augmentation:
        trc_files = [f for f in pose3d_dir.glob('*.trc') if '_LSTM' in f.name]
        if len(trc_files) == 0:
            pose_model = 'HALPE_26'
            use_augmentation = False
            logger.warning("No LSTM trc files found. Using non augmented trc files instead.")
    if len(trc_files) == 0: # filtered files by default
        trc_files = [f for f in pose3d_dir.glob('*.trc') if '_LSTM' not in f.name and '_filt' in f.name and '_scaling' not in f.name]
    if len(trc_files) == 0: 
        trc_files = [f for f in pose3d_dir.glob('*.trc') if '_LSTM' not in f.name and '_scaling' not in f.name]
    if len(trc_files) == 0:
        raise ValueError(f'No trc files found in {pose3d_dir}.')
    trc_files = sorted(trc_files, key=natural_sort_key)

    pose_model = 'LSTM' if use_augmentation else 'HALPE_26'
    subject_height = [subject_height] if not isinstance(subject_height, list) else subject_height
    subject_mass = [subject_mass] if not isinstance(subject_mass, list) else subject_mass
    
    # Perform scaling and IK for each trc file
    for p, trc_file in enumerate(trc_files):
        _log(f"Processing TRC file: {trc_file.resolve()}")

        _log("\nScaling...")
        perform_scaling(trc_file, pose_model, kinematics_dir, osim_setup_dir, use_simple_model, right_left_symmetry=right_left_symmetry, subject_height=subject_height[p], subject_mass=subject_mass[p],
                        remove_scaling_setup=remove_scaling_setup, fastest_frames_to_remove_percent=fastest_frames_to_remove_percent, large_hip_knee_angles=large_hip_knee_angles, trimmed_extrema_percent=trimmed_extrema_percent,close_to_zero_speed_m=close_to_zero_speed)
        _log(f"\tDone. OpenSim logs saved to {opensim_logs_file.resolve()}.")
        _log(f"\tScaled model saved to {(kinematics_dir / (trc_file.stem + '_scaled.osim')).resolve()}")

        _log("\nInverse Kinematics...")
        start_time = time.time()
        perform_IK(trc_file, kinematics_dir, osim_setup_dir, pose_model, remove_IK_setup=remove_IK_setup)
        end_time = time.time()
        _log(f"\tIK took {round(end_time - start_time, 2)} seconds for {trc_file.name}.")
        _log(f"\tDone. OpenSim logs saved to {opensim_logs_file.resolve()}.")
        mot_path = kinematics_dir / (trc_file.stem + '.mot')
        _log(f"\tJoint angle data saved to {mot_path.resolve()}")
        keypoint_trc_path = resolve_keypoint_trc(Path(project_dir), trc_file)
        combined_csv_path = export_combined_kinematics_csv(
            Path(project_dir),
            mot_path,
            keypoint_trc_path,
            kinematics_filter,
            subject_metadata=sub,
            fps=base.get('fps'),
        )
        _log(f"\tCombined keypoint and kinematics CSV saved to {combined_csv_path.resolve()}\n")
