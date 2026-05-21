"""TRC 파일 읽기/쓰기 유틸리티."""
import numpy as np
import pandas as pd

def read_trc(trc_path):
    """
    TRC 파일을 읽어 내용을 반환한다.

    Returns
    -------
    tuple: (Q_coords, frames_col, time_col, markers, header)
    """
    try:
        with open(trc_path, 'r') as trc_file:
            header = [next(trc_file) for _ in range(5)]
        markers = header[3].split('\t')[2::3]
        markers = [m.strip() for m in markers if m.strip()]

        trc_df = pd.read_csv(trc_path, sep="\t", skiprows=4, encoding='utf-8')
        frames_col = trc_df.iloc[:, 0]
        time_col = trc_df.iloc[:, 1]
        Q_coords = trc_df.drop(trc_df.columns[[0, 1]], axis=1)
        Q_coords = Q_coords.loc[:, ~Q_coords.columns.str.startswith('Unnamed')]
        Q_coords.columns = np.array([[m, m, m] for m in markers]).ravel().tolist()

        return Q_coords, frames_col, time_col, markers, header

    except Exception as e:
        raise ValueError(f"Error reading TRC file at {trc_path}: {e}")

    
def export_to_trc(filename, optimized_3d, valid_frames, fps, TRC_MARKERS, out_unit='m'):
    num_frames = len(valid_frames)
    num_markers = len(TRC_MARKERS)

    # TRC는 m 단위를 사용하므로 입력 3D 단위를 m로 변환
    unit_scale_to_m = 1.0
    if out_unit == 'mm':
        unit_scale_to_m = 1.0 / 1000.0
    elif out_unit == 'cm':
        unit_scale_to_m = 1.0 / 100.0
    elif out_unit == 'm':
        unit_scale_to_m = 1.0
    else:
        # 알 수 없는 단위면 기존 동작(=cm 가정)으로 처리
        unit_scale_to_m = 1.0 / 100.0

    with open(filename, 'w') as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{filename}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps}\t{fps}\t{num_frames}\t{num_markers}\tm\t{fps}\t{valid_frames[0]}\t{valid_frames[-1]}\n")
        f.write("Frame#\tTime\t")
        for name, _ in TRC_MARKERS: f.write(f"{name}\t\t\t")
        f.write("\n\t\t")
        for i in range(num_markers): f.write(f"X{i+1}\tY{i+1}\tZ{i+1}\t")
        f.write("\n\n")
        
        for i, original_frame_idx in enumerate(valid_frames):
            time = original_frame_idx / fps
            f.write(f"{original_frame_idx}\t{time:.4f}\t")
            for _, joint_idx in TRC_MARKERS:
                x, y, z = optimized_3d[i, joint_idx] * unit_scale_to_m
                f.write(f"{x:.4f}\t{y:.4f}\t{z:.4f}\t")
            f.write("\n")
