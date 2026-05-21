import os
import glob
import numpy as np
np.set_printoptions(legacy='1.21') # otherwise prints np.float64(3.0) rather than 3.0
import logging

from scipy import signal
from .utilities import read_trc

logger = logging.getLogger(__name__)

def hampel_filter(col, window_size=7, n_sigma=2):
    '''
    Hampel filter for outlier rejection before other filtering methods.
    Takes a sliding window of size 7, calculates its median and standard deviation, 
    replaces value by median if difference is more than 2 times the standard deviation (95% confidence interval), 
    else keeps the value.
    '''

    col_filtered = col.copy()
    half_window = window_size // 2
    
    for i in range(half_window, len(col) - half_window):
        window = col[i-half_window:i+half_window+1]
        val = col[i]

        # NaN이 섞이면 median/mad가 NaN이 되어 필터가 깨질 수 있어
        # window의 finite 값만으로 robust statistics를 계산한다.
        if not np.isfinite(val):
            continue

        window_vals = window[np.isfinite(window)]
        if window_vals.size == 0:
            continue

        median = np.median(window_vals)
        mad = np.median(np.abs(window_vals - median))  # Median Absolute Deviation

        if mad > 0 and np.isfinite(mad):
            modified_z_score = 0.6745 * (val - median) / mad #75% percentile from median
            if np.abs(modified_z_score) > n_sigma:
                col_filtered[i] = median
    
    return col_filtered

def butterworth_filter_1d(config_dict, frame_rate, col):
    '''
    1D Zero-phase Butterworth filter (dual pass)
    Deals with nans

    INPUT:
    - col: numpy array
    - order: int
    - cutoff: int
    - frame_rate: int

    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''

    type = 'low' #config_dict.get('filtering').get('butterworth').get('type')
    order = int(config_dict.get('filtering').get('butterworth').get('order'))
    cutoff = int(config_dict.get('filtering').get('butterworth').get('cut_off_frequency'))    

    b, a = signal.butter(order/2, cutoff/(frame_rate/2), type, analog = False) 
    padlen = 3 * max(len(a), len(b))
    
    # split into sequences of not nans
    col_filtered = col.copy()
    mask = np.isnan(col_filtered)  | col_filtered.eq(0)
    falsemask_indices = np.where(~mask)[0]
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1 
    idx_sequences = np.split(falsemask_indices, gaps)
    if idx_sequences[0].size > 0:
        idx_sequences_to_filter = [seq for seq in idx_sequences if len(seq) > padlen]
    
        # Filter each of the selected sequences
        for seq_f in idx_sequences_to_filter:
            col_filtered[seq_f] = signal.filtfilt(b, a, col_filtered[seq_f])
    
    return col_filtered

def filter1d(col, config_dict, filter_type, frame_rate):
    '''
    Choose filter type and filter column

    INPUT:
    - col: Pandas dataframe column
    - filter_type: filter type from Config.toml
    - frame_rate: int
    
    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''
    # Choose filter
    filter_mapping = {
        'butterworth': butterworth_filter_1d
        }
    filter_fun = filter_mapping[filter_type]
    
    # Filter column
    col_filtered = filter_fun(config_dict, frame_rate, col)

    return col_filtered

def recap_filter3d(config_dict, trc_path):
    '''
    Print a log message giving filtering parameters. Also stored in User/logs.txt.

    OUTPUT:
    - Message in console
    '''

    # Read Config
    butterworth_filter_order = int(config_dict.get('filtering').get('butterworth').get('order'))
    butterworth_filter_cutoff = int(config_dict.get('filtering').get('butterworth').get('cut_off_frequency'))
    
    # Recap
    logger.info('--> Outliers rejected with a Hampel filter.')
    logger.info(f'--> Filter type: Butterworth pass. Order {butterworth_filter_order}, Cut-off frequency {butterworth_filter_cutoff} Hz.')
    logger.info(f'Filtered 3D coordinates are stored at {trc_path}.')
        
def run_filtering(config_dict, emit_log=None):
    '''
    Filter the 3D coordinates of the trc file.
    Displays filtered coordinates for checking.

    INPUTS:
    - a trc file
    - filtration parameters from Config.toml

    OUTPUT:
    - a filtered trc file
    '''

    def _log(text, level='info'):
        if callable(emit_log):
            emit_log(text, level)
        else:
            logger.info(text)

    # Read config_dict
    project_dir = config_dict.get('paths').get('project_dir')
    pose3d_dir = os.path.realpath(os.path.join(project_dir, 'pose-3d'))
    hampel_window_size = config_dict.get('filtering').get('hampel').get('window_size')
    hampel_n_sigma = config_dict.get('filtering').get('hampel').get('n_sigma')
    # hampel_interp_limit = config_dict.get('filtering').get('hampel').get('interp_limit')
    filter_type = 'butterworth'
    
    # Get frame_rate
    frame_range = config_dict.get('base').get('frame_range')
    frame_rate = config_dict.get('base').get('fps')
    
    # Trc paths
    trc_path_in = [file for file in glob.glob(os.path.join(pose3d_dir, '*.trc')) if 'filt' not in file]
    for person_id, t_path_in in enumerate(trc_path_in):
        _log('--> Filtering 3D coordinates...')
        # Read trc coordinate values
        t_file_in = os.path.basename(t_path_in)
        Q_coords, frames_col, time_col, markers, header = read_trc(t_path_in)

        # frame range selection
        if len(frames_col) == 0:
            logger.error("--> Filtering skipped: empty frames_col in TRC.")
            continue

        f_range = (
            [[frames_col.iloc[0], frames_col.iloc[-1] + 1] if frame_range in ('all', 'auto', []) else frame_range][0]
        )
        # frame_range는 [start, end)로 취급 (end 미포함)
        target_start = int(f_range[0])
        target_end_incl = int(f_range[1] - 1)

        # frames_col에는 특정 frame 값이 없을 수 있어(동기화/크롭/누락),
        # == 조건으로 직접 index를 찾으면 out-of-bounds가 나올 수 있다.
        start_candidates = frames_col[frames_col >= target_start].index
        end_candidates = frames_col[frames_col <= target_end_incl].index

        if start_candidates.size == 0 or end_candidates.size == 0:
            # 요청 구간이 TRC에 없더라도 파이프라인이 중단되지 않도록 폴백
            logger.error(
                "--> Filtering frame_range not found in TRC; fallback to full available frames. "
                f"requested=[{target_start}, {target_end_incl}] TRC=[{int(frames_col.min())}, {int(frames_col.max())}]"
            )
            start_i = 0
            end_i_excl = len(frames_col)
        else:
            start_i = int(start_candidates[0])
            end_i_excl = int(end_candidates[-1] + 1)

        if end_i_excl <= start_i:
            logger.error("--> Filtering skipped: invalid sliced indices after frame_range adjustment.")
            continue

        Q_coords = Q_coords.iloc[start_i:end_i_excl].reset_index(drop=True)
        frames_col = frames_col.iloc[start_i:end_i_excl].reset_index(drop=True)
        time_col = time_col.iloc[start_i:end_i_excl].reset_index(drop=True)

        # 사람 미검출(안 보임) 프레임은 보통 모든 마커 좌표가 NaN.
        # 그런 프레임을 drop하고 필터를 적용해 NaN 전파/빈 데이터 크래시를 방지한다.
        q_vals = Q_coords.to_numpy()
        valid_row_mask = np.isfinite(q_vals).any(axis=1)
        if not valid_row_mask.any():
            logger.error("--> Filtering skipped: all rows are NaN after slicing.")
            continue
        if (~valid_row_mask).any():
            dropped = int((~valid_row_mask).sum())
            logger.warning(f"--> Dropping {dropped} frames where all 3D coordinates are NaN.")
            Q_coords = Q_coords.loc[valid_row_mask].reset_index(drop=True)
            frames_col = frames_col.loc[valid_row_mask].reset_index(drop=True)
            time_col = time_col.loc[valid_row_mask].reset_index(drop=True)

        frame_nb = int(len(frames_col))

        t_path_out = os.path.join(pose3d_dir, f'keypoints_3d_filt_{filter_type}.trc')
        t_file_out = os.path.basename(t_path_out)
        header[0] = header[0].replace(t_file_in, t_file_out)
        header[2] = '\t'.join(part if i != 2 else str(frame_nb) for i, part in enumerate(header[2].split('\t')))
        header[2] = '\t'.join(part if i != 7 else str(frame_nb)+'\n' for i, part in enumerate(header[2].split('\t')))

        # Filter coordinates
        Q_coords = Q_coords.apply(hampel_filter, axis=0, args = [hampel_window_size, hampel_n_sigma])  # Hampel filter for outlier rejection
        Q_filt = Q_coords.apply(filter1d, axis=0, args = [config_dict, filter_type, frame_rate])
        
        # Reconstruct trc file with filtered coordinates
        with open(t_path_out, 'w') as trc_o:
            [trc_o.write(line) for line in header]
            Q_filt.insert(0, 'Frame#', frames_col)
            Q_filt.insert(1, 'Time', time_col)
            # Q_filt = Q_filt.fillna(' ')
            Q_filt.to_csv(trc_o, sep='\t', index=False, header=None, lineterminator='\n')

        # Recap
        recap_filter3d(config_dict, t_path_out)
