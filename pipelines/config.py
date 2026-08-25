"""분석 파이프라인 설정 — dataclass 기반.

기존 dict 파이프라인과의 호환을 위해 to_dict() 메서드를 제공한다.
서비스 레이어에서는 dataclass로 타입 안전하게 다루고,
파이프라인 내부(analyzing.py 등)에 전달 시 to_dict()로 변환한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── 하위 설정 dataclass ────────────────────────────────────────────────────────

@dataclass
class BaseConfig:
    motion: str = "walking"
    walking_direction: str = "-z"
    frame_range: str = "auto"  # "auto" 또는 [start, end]; 변경하지 말 것


@dataclass
class PoseConfig:
    mode: str = "normal"                        # normal | performance
    device: str = "auto"                        # 변경 X
    backend: str = "auto"                       # 변경 X
    det_score_threshold: float = 0.25
    det_iou: float = 0.7
    det_nms: bool = True
    keypoint_likelihood_threshold: float = 0.5
    average_likelihood_threshold: float = 0.5
    keypoint_number_threshold: float = 0.3
    output_format: str = "openpose"             # 변경 X
    save_video: bool = True                     # 변경 X
    overwrite_pose: bool = False
    max_distance_px: int = 150


@dataclass
class LiftingConfig:
    feet_on_floor: bool = True
    flip_left_right: bool = False
    
    # camera calibration file 있는 경우 세팅
    reproj_error_threshold_triangulation: float = 15 # px # if reprojection error is above, triangulation results won't be accepted
    likelihood_threshold_triangulation: float = 0.3 # if 2D likelihood is below, estimations for this camera won't be accepted
    min_cameras_for_triangulation: int = 2 # stops trying to triangulate if results still not good with min_cameras_for_triangulation cameras. 
                                    # Increase if you have many cameras and want to be more robust to occlusions
    max_distance_m: float = 1.0 # m # max distance a person can jump from their previous position before being considered as a new one
    max_unseen_frames: float = 100 # max number of frames that a person can be unseen before the next person passing by is given a new ID
    interp_if_gap_smaller_than: int = 20 # do not interpolate larger gaps
    interpolation: str = 'linear' # linear, slinear, quadratic, cubic, or none # 'none' if you don't want to interpolate missing points
    remove_incomplete_frames: bool = False # true or false (lowercase) # If true, frame is kept only if all keypoints have been correctly triangulated
    sections_to_keep: str = 'all' # 'all', 'largest', 'first', 'last'
                            # keep 'all' correctly triangulated sections, or the 'largest' valid section, or the 'first' one, or the 'last' one
    min_chunk_size: int = 10 # Minimum number of consecutive valid frames to make it a section to keep
    fill_large_gaps_with: str = 'last_value' # 'last_value', 'nan', or 'zeros'
    show_interp_indices: bool = True # true or false (lowercase). For each keypoint, return the frames that need to be interpolated                         
     
    


@dataclass
class HampelConfig:
    window_size: int = 7
    n_sigma: int = 2
    interp_limit: int = 15


@dataclass
class ButterworthConfig:
    order: int = 4
    cut_off_frequency: float = 6.0


@dataclass
class FilteringConfig:
    hampel: HampelConfig = field(default_factory=HampelConfig)
    butterworth: ButterworthConfig = field(default_factory=ButterworthConfig)


@dataclass
class KinematicsConfig:
    use_simple_model: bool = True
    use_augmentation: bool = True
    right_left_symmetry: bool = True
    remove_individual_scaling_setup: bool = True
    remove_individual_ik_setup: bool = True
    fastest_frames_to_remove_percent: float = 0.1
    close_to_zero_speed_m: float = 0.2
    large_hip_knee_angles: float = 135.0
    trimmed_extrema_percent: float = 0.5
    filter_cut_off_frequency: float = 6.0
    filter_order: int = 4


# ── 루트 설정 dataclass ────────────────────────────────────────────────────────

@dataclass
class AutoCalibrationConfig:
    """Used only when a session has no calibration file of its own."""

    frame_stride: int = 5          # sample every Nth pose frame
    max_frames: int = 300          # upper bound on sampled frames
    conf_threshold: float = 0.5    # minimum keypoint confidence to trust a joint
    ba_iterations: int = 2         # bundle adjustment passes after the linear solve
    focal_ratio: float = 0.9       # focal length seed as a fraction of image width


@dataclass
class AnalysisConfig:
    base: BaseConfig = field(default_factory=BaseConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    auto_calibration: AutoCalibrationConfig = field(default_factory=AutoCalibrationConfig)
    lifting: LiftingConfig = field(default_factory=LiftingConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)

    def to_dict(self) -> dict:
        """기존 dict 기반 파이프라인과 호환되는 dict를 반환한다."""
        return {
            "base": {
                "motion": self.base.motion,
                "walking_direction": self.base.walking_direction,
                "frame_range": self.base.frame_range,
            },
            "pose": {
                "mode": self.pose.mode,
                "device": self.pose.device,
                "backend": self.pose.backend,
                "det_score_threshold": self.pose.det_score_threshold,
                "det_iou": self.pose.det_iou,
                "det_nms": self.pose.det_nms,
                "keypoint_likelihood_threshold": self.pose.keypoint_likelihood_threshold,
                "average_likelihood_threshold": self.pose.average_likelihood_threshold,
                "keypoint_number_threshold": self.pose.keypoint_number_threshold,
                "output_format": self.pose.output_format,
                "save_video": self.pose.save_video,
                "overwrite_pose": self.pose.overwrite_pose,
                "max_distance_px": self.pose.max_distance_px,
            },
            "auto_calibration": {
                "frame_stride": self.auto_calibration.frame_stride,
                "max_frames": self.auto_calibration.max_frames,
                "conf_threshold": self.auto_calibration.conf_threshold,
                "ba_iterations": self.auto_calibration.ba_iterations,
                "focal_ratio": self.auto_calibration.focal_ratio,
            },
            "lifting": {
                "flip_left_right": self.lifting.flip_left_right,
                "feet_on_floor": self.lifting.feet_on_floor,
                "reproj_error_threshold_triangulation": self.lifting.reproj_error_threshold_triangulation,
                "likelihood_threshold_triangulation": self.lifting.likelihood_threshold_triangulation,
                "min_cameras_for_triangulation": self.lifting.min_cameras_for_triangulation,
                "max_distance_m": self.lifting.max_distance_m,
                "max_unseen_frames": self.lifting.max_unseen_frames,
                "interp_if_gap_smaller_than": self.lifting.interp_if_gap_smaller_than,
                "interpolation": self.lifting.interpolation,
                "remove_incomplete_frames": self.lifting.remove_incomplete_frames,
                "sections_to_keep": self.lifting.sections_to_keep,
                "min_chunk_size": self.lifting.min_chunk_size,
                "fill_large_gaps_with": self.lifting.fill_large_gaps_with,
                "show_interp_indices": self.lifting.show_interp_indices,
            },
            "filtering": {
                "hampel": {
                    "window_size": self.filtering.hampel.window_size,
                    "n_sigma": self.filtering.hampel.n_sigma,
                    "interp_limit": self.filtering.hampel.interp_limit,
                },
                "butterworth": {
                    "order": self.filtering.butterworth.order,
                    "cut_off_frequency": self.filtering.butterworth.cut_off_frequency,
                },
            },
            "kinematics": {
                "use_simple_model": self.kinematics.use_simple_model,
                "use_augmentation": self.kinematics.use_augmentation,
                "right_left_symmetry": self.kinematics.right_left_symmetry,
                "remove_individual_scaling_setup": self.kinematics.remove_individual_scaling_setup,
                "remove_individual_ik_setup": self.kinematics.remove_individual_ik_setup,
                "fastest_frames_to_remove_percent": self.kinematics.fastest_frames_to_remove_percent,
                "close_to_zero_speed_m": self.kinematics.close_to_zero_speed_m,
                "large_hip_knee_angles": self.kinematics.large_hip_knee_angles,
                "trimmed_extrema_percent": self.kinematics.trimmed_extrema_percent,
                "filter": {
                    "cut_off_frequency": self.kinematics.filter_cut_off_frequency,
                    "order": self.kinematics.filter_order,
                },
            },
        }

    @staticmethod
    def defaults() -> "AnalysisConfig":
        """기본값으로 채워진 AnalysisConfig 인스턴스를 반환한다.

        파이프라인 내부에서 DEFAULT_CONFIG가 필요한 곳은
        ``AnalysisConfig.defaults().to_dict()`` 를 사용한다.
        """
        return AnalysisConfig()

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisConfig":
        """dict에서 AnalysisConfig를 생성한다. 없는 키는 기본값으로 채운다."""
        base_d = d.get("base") or {}
        pose_d = d.get("pose") or {}
        auto_d = d.get("auto_calibration") or {}
        lift_d = d.get("lifting") or {}
        filt_d = d.get("filtering") or {}
        kin_d = d.get("kinematics") or {}
        hamp_d = filt_d.get("hampel") or {}
        butt_d = filt_d.get("butterworth") or {}
        kin_filter_d = kin_d.get("filter") or {}

        return cls(
            auto_calibration=AutoCalibrationConfig(
                frame_stride=int(auto_d.get("frame_stride", 5)),
                max_frames=int(auto_d.get("max_frames", 300)),
                conf_threshold=float(auto_d.get("conf_threshold", 0.5)),
                ba_iterations=int(auto_d.get("ba_iterations", 2)),
                focal_ratio=float(auto_d.get("focal_ratio", 0.9)),
            ),
            base=BaseConfig(
                motion=base_d.get("motion", "Baseball-Pitching"),
                walking_direction=base_d.get("walking_direction", "-z"),
                frame_range=base_d.get("frame_range", "auto"),
            ),
            pose=PoseConfig(
                mode=pose_d.get("mode", "normal"),
                device=pose_d.get("device", "auto"),
                backend=pose_d.get("backend", "auto"),
                det_score_threshold=float(pose_d.get("det_score_threshold", 0.25)),
                det_iou=float(pose_d.get("det_iou", 0.7)),
                det_nms=bool(pose_d.get("det_nms", True)),
                keypoint_likelihood_threshold=float(pose_d.get("keypoint_likelihood_threshold", 0.5)),
                average_likelihood_threshold=float(pose_d.get("average_likelihood_threshold", 0.5)),
                keypoint_number_threshold=float(pose_d.get("keypoint_number_threshold", 0.3)),
                output_format=pose_d.get("output_format", "openpose"),
                save_video=bool(pose_d.get("save_video", True)),
                overwrite_pose=bool(pose_d.get("overwrite_pose", False)),
                max_distance_px=int(pose_d.get("max_distance_px", 150)),
            ),
            lifting=LiftingConfig(
                flip_left_right=bool(lift_d.get("flip_left_right", False)),
                feet_on_floor=bool(lift_d.get("feet_on_floor", True)),
                reproj_error_threshold_triangulation=float(lift_d.get("reproj_error_threshold_triangulation", 15)),
                likelihood_threshold_triangulation=float(lift_d.get("likelihood_threshold_triangulation", 0.3)),
                min_cameras_for_triangulation=int(lift_d.get("min_cameras_for_triangulation", 2)),
                max_distance_m=float(lift_d.get("max_distance_m", 1.0)),
                max_unseen_frames=float(lift_d.get("max_unseen_frames", 100)),
                interp_if_gap_smaller_than=int(lift_d.get("interp_if_gap_smaller_than", 20)),
                interpolation=lift_d.get("interpolation", "linear"),
                remove_incomplete_frames=bool(lift_d.get("remove_incomplete_frames", False)),
                sections_to_keep=lift_d.get("sections_to_keep", "all"),
                min_chunk_size=int(lift_d.get("min_chunk_size", 10)),
                fill_large_gaps_with=lift_d.get("fill_large_gaps_with", "last_value"),
                show_interp_indices=bool(lift_d.get("show_interp_indices", True)),
            ),
            filtering=FilteringConfig(
                hampel=HampelConfig(
                    window_size=int(hamp_d.get("window_size", 7)),
                    n_sigma=int(hamp_d.get("n_sigma", 2)),
                    interp_limit=int(hamp_d.get("interp_limit", 15)),
                ),
                butterworth=ButterworthConfig(
                    order=int(butt_d.get("order", 4)),
                    cut_off_frequency=float(butt_d.get("cut_off_frequency", 6.0)),
                ),
            ),
            kinematics=KinematicsConfig(
                use_simple_model=bool(kin_d.get("use_simple_model", True)),
                use_augmentation=bool(kin_d.get("use_augmentation", True)),
                right_left_symmetry=bool(kin_d.get("right_left_symmetry", True)),
                remove_individual_scaling_setup=bool(kin_d.get("remove_individual_scaling_setup", True)),
                remove_individual_ik_setup=bool(kin_d.get("remove_individual_ik_setup", True)),
                fastest_frames_to_remove_percent=float(kin_d.get("fastest_frames_to_remove_percent", 0.1)),
                close_to_zero_speed_m=float(kin_d.get("close_to_zero_speed_m", 0.2)),
                large_hip_knee_angles=float(kin_d.get("large_hip_knee_angles", 135.0)),
                trimmed_extrema_percent=float(kin_d.get("trimmed_extrema_percent", 0.5)),
                filter_cut_off_frequency=float(kin_filter_d.get("cut_off_frequency", 6.0)),
                filter_order=int(kin_filter_d.get("order", 4)),
            ),
        )


# ── 모듈 수준 기본값 — 파이프라인에서 deep_merge(DEFAULT_CONFIG, user_config) 패턴으로 사용 ──

DEFAULT_CONFIG: dict = AnalysisConfig().to_dict()


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    """base 에 override 를 재귀적으로 덮어쓴 새 dict 를 반환한다."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result
