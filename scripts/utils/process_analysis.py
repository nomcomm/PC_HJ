import numpy as np
import pandas as pd
import os
import glob
from scipy.stats import zscore
from nltools.stats import isc
from scipy.stats import pearsonr


def get_subjects_by_condition(conditions_file, target_condition='PC', exclude_prior_exposure=False):
    """
    Reads the conditions file and returns a list of subject IDs for the target condition.
    
    Parameters:
    - exclude_prior_exposure (bool): If True, removes subjects where 'Seen' == 1.
    """
    if not os.path.exists(conditions_file):
        raise FileNotFoundError(f"Conditions file not found: {conditions_file}")
        
    df = pd.read_csv(conditions_file)
    
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    
    # Check required columns
    required = ['RespondentName', 'SourceStimulus']
    if not set(required).issubset(df.columns):
        raise ValueError(f"Conditions file must contain {required} columns.")
    
    # 1. Filter by Target Condition first
    condition_df = df[df['SourceStimulus'].str.strip() == target_condition].copy()
    
    # 2. Filter by Prior Exposure (New Logic)
    if exclude_prior_exposure:
        if 'Seen' in condition_df.columns:
            # Get initial count for reporting
            n_before = len(condition_df)
            
            # Filter: Keep rows where Seen is NOT 1 (handles 0, NaN, or empty)
            # We convert to numeric to be safe against "1" (str) vs 1 (int)
            condition_df['Seen'] = pd.to_numeric(condition_df['Seen'], errors='coerce').fillna(0)
            condition_df = condition_df[condition_df['Seen'] != 1]
            
            n_excluded = n_before - len(condition_df)
            if n_excluded > 0:
                print(f"[{target_condition}] Excluded {n_excluded} subjects due to prior exposure.")
        else:
            print(f"Warning: 'exclude_prior_exposure' was True, but 'Seen' column was not found in {conditions_file}.")

    # 3. Extract IDs
    subject_ids = condition_df['RespondentName'].apply(parse_subject_id).tolist()
    
    # Remove duplicates and sort
    subject_ids = sorted(list(set(subject_ids)))
    
    return subject_ids

def parse_subject_id(respondent_name):
    """
    Extracts subject ID (e.g., 'sub001') from RespondentName string.
    Handles potential extra spaces or underscores.
    """
    # Split by underscore and take the first part
    parts = respondent_name.split('_')
    sub_id = parts[0].strip()
    return sub_id

def load_filtered_physiological_data(folder_path, subject_list):
    """
    Loads CSV files ONLY if they match an ID in subject_list.
    Checks for required columns and data integrity.
    Returns a dictionary of DataFrames (Time x Subjects).
    """
    
    required_cols = ['Time_sec', 'GSR', 'Heart_Rate', 'Pupil_Mean', 'Gaze_X', 'Gaze_Y']
    signal_types = ['GSR', 'Heart_Rate', 'Pupil_Mean', 'Gaze_X', 'Gaze_Y']
    
    # 1. Structure to hold data: {'GSR': [], 'Heart_Rate': [], ...}
    data_collector = {signal: [] for signal in signal_types}
    
    file_list = glob.glob(os.path.join(folder_path, "*.csv"))
    
    print(f"Scanning {len(file_list)} files against {len(subject_list)} subject IDs...")
    
    loaded_count = 0
    
    for filepath in file_list:
        filename = os.path.basename(filepath)
        
        # --- FILTER 1: CHECK FILENAME AGAINST LIST ---
        # We look for a subject ID that exists inside this filename.
        # e.g., if filename is 'sub003_combined_physio.csv' and 'sub003' is in the list
        matched_subject_id = None
        
        for subj_id in subject_list:
            if subj_id in filename:
                matched_subject_id = subj_id
                break # Found the match, stop checking other IDs
        
        # If this file doesn't match anyone in our list, skip it silently
        if matched_subject_id is None:
            continue

        try:
            # Load data
            df = pd.read_csv(filepath)
            
            # --- FILTER 2: CHECK COLUMN EXISTENCE ---
            if not set(required_cols).issubset(df.columns):
                print(f"Skipping {matched_subject_id} (File: {filename}) - Missing required columns")
                continue
            
            # --- FILTER 3: CHECK FOR EMPTY/NAN DATA ---
            # Check if any required column has NaN values
            if df[required_cols].isnull().any().any():
                print(f"Skipping {matched_subject_id} (File: {filename}) - Contains NaNs/Missing data")
                continue
                
            # --- PROCESSING ---
            # 1. Handle Time Index (Round to 3 decimals to ensure alignment across subjects)
            df['Time_sec'] = df['Time_sec'].round(3)
            df = df.set_index('Time_sec')
            
            # 2. Add each signal to its respective list
            for signal in signal_types:
                s = df[signal]
                s.name = matched_subject_id  # Column name becomes 'sub002', etc.
                data_collector[signal].append(s)
                
            loaded_count += 1
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print(f"Successfully loaded {loaded_count} valid subjects.")

    # --- MERGING ---
    final_dfs = {}
    for signal, series_list in data_collector.items():
        if series_list:
            # Concat aligns on the Index (Time_sec) automatically
            merged_df = pd.concat(series_list, axis=1)
            merged_df.sort_index(inplace=True)
            final_dfs[signal] = merged_df
        else:
            final_dfs[signal] = pd.DataFrame()

    return final_dfs

def get_zscored_split_half(data_df):
    """
    Creates ONE fixed split-half for the timecourse plot.
    Z-scores them so they share a visual scale.
    """
    data = data_df.values
    n_subjects = data.shape[1]
    
    # Use fixed seed for reproducibility of the plot
    np.random.seed(42) 
    indices = np.arange(n_subjects)
    np.random.shuffle(indices)
    mid = n_subjects // 2
    
    g1 = data[:, indices[:mid]].mean(axis=1)
    g2 = data[:, indices[mid:]].mean(axis=1)
    
    # Z-score
    g1_z = zscore(g1, nan_policy='omit')
    g2_z = zscore(g2, nan_policy='omit')
    
    return g1_z, g2_z, np.corrcoef(g1_z, g2_z)[0, 1]
    
    

# --- HELPER: PERMUTATION STATS ---
def calculate_permuted_stats(df_a, df_b, n_permutations=5000):
    """
    Returns: Mean r, Low_Err (distance), High_Err (distance)
    """
    n_a, n_b = df_a.shape[1], df_b.shape[1]
    k_a, k_b = int(n_a // 2), int(n_b // 2)
    
    if k_a < 1 or k_b < 1: return 0, 0, 0
    
    r_values = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        idx_a = np.random.choice(n_a, k_a, replace=False)
        idx_b = np.random.choice(n_b, k_b, replace=False)
        g1 = df_a.iloc[:, idx_a].mean(axis=1)
        g2 = df_b.iloc[:, idx_b].mean(axis=1)
        r = np.corrcoef(g1, g2)[0, 1]
        r_values[i] = np.clip(r, -0.999999, 0.999999)

    # Fisher Z Statistics
    z_values = np.arctanh(r_values)
    mean_z = np.mean(z_values)
    ci_lower_z = np.percentile(z_values, 2.5)
    ci_upper_z = np.percentile(z_values, 97.5)
    
    mean_r = np.tanh(mean_z)
    # Return distances for error bars
    return mean_r, (mean_r - np.tanh(ci_lower_z)), (np.tanh(ci_upper_z) - mean_r)



def load_physio_data(data_dir, subject_ids, signal_type='GSR'):
    """
    Loads physiological data for a list of subjects.
    
    Args:
        data_dir (str): Path to processed combined data.
        subject_ids (list): List of subject IDs to load (e.g. ['sub001', 'sub002']).
        signal_type (str): 'GSR' or 'Heart_Rate'.
        
    Returns:
        pd.DataFrame: DataFrame with shape (n_timepoints, n_subjects).
                      Columns are subject IDs.
    """
    aligned_data = {}
    valid_subjects = []
    
    # Find all files first to handle case insensitivity (sub vs Sub)
    # We look for both '_combined_physio.csv' AND '_processed_et.csv' or similar depending on where ET is.
    # Actually, previous conversation said "Integrate Physiological Data" into combined csv.
    # Let's check if Gaze/Pupil columns are in the SAME combined file or different files.
    # The 'sub001_combined_physio.csv' we viewed earlier HAD 'Pupil_Mean', 'Gaze_X', 'Gaze_Y'.
    # So we just need to ensure we map those column names correctly.
    
    all_files = glob.glob(os.path.join(data_dir, '*combined_physio.csv'))
    file_map = {}
    for f in all_files:
        basename = os.path.basename(f)
        file_id = basename.split('_')[0].lower()
        file_map[file_id] = f
        
    lengths = []
    
    for sub_id in subject_ids:
        # Check if we have a file for this subject
        lookup_id = sub_id.lower()
        
        if lookup_id not in file_map:
            #print(f"  Warning: No data file found for subject {sub_id}")
            continue
            
        file_path = file_map[lookup_id]
        
        try:
            df = pd.read_csv(file_path)
            
            # Column Mapping fallback
            col_map = {
                'GSR': ['GSR', 'GSR_Interpolated'],
                'Heart_Rate': ['Heart_Rate', 'Heart Rate'],
                'Gaze_X': ['Gaze_X', 'Interpolated Gaze X', 'X'],
                'Gaze_Y': ['Gaze_Y', 'Interpolated Gaze Y', 'Y'],
                'Pupil_Mean': ['Pupil_Mean', 'Mean_Pupil', 'Pupil Diameter']
            }
            
            target_col = signal_type
            if signal_type in col_map:
                for possible in col_map[signal_type]:
                    if possible in df.columns:
                        target_col = possible
                        break
            
            if target_col not in df.columns:
                print(f"  Warning: Column '{signal_type}' (or variants) not found in {os.path.basename(file_path)}")
                continue
            
            signal_data = df[target_col].values
            
            # Basic cleaning: handle infinite or NaN
            if not np.isfinite(signal_data).all():
                # Interpolate linear
                signal_series = pd.Series(signal_data)
                # First interpolate internal holes
                signal_series = signal_series.interpolate(method='linear', limit_direction='both')
                # Then fill edges just in case
                signal_series = signal_series.ffill().bfill()
                signal_data = signal_series.values
                
            # If still NaNs (e.g. all NaNs), warning or skip
            if np.isnan(signal_data).any():
                #print(f"  Warning: {sub_id} {target_col} contains NaNs after interpolation. Dropping.")
                continue
                
            aligned_data[sub_id] = signal_data
            valid_subjects.append(sub_id)
            lengths.append(len(signal_data))
            
        except Exception as e:
            print(f"  Error loading {sub_id}: {e}")
            
    if not aligned_data:
        raise ValueError("No valid data loaded for any subject!")
        
    # Determine common length
    # We use the minimum length to ensure no NaNs at the end, 
    # assuming start times are aligned (which they should be from previous processing).
    # However, if min length is too short (bad subject), we might want to exclude that subject.
    # Let's verify lengths.
    
    # Histogram of lengths
    length_counts = pd.Series(lengths).value_counts().sort_index()
    #print(f"\nData Length Distribution for {signal_type}:")
    #print(length_counts)
    
    # Use the most common length or min length?
    # If we use min, we might cut good data because of one short file.
    # If we use mode, we might need to pad or trim.
    # Let's use the minimum length that is "reasonable" (e.g. at least 90% of median).
    median_len = np.median(lengths)
    threshold = 0.9 * median_len
    
    # Filter out subjects that are too short
    final_subjects = []
    clean_data_dict = {}
    
    valid_lengths = [l for l in lengths if l >= threshold]
    if not valid_lengths:
         raise ValueError("All loaded files are too short compared to median length!")
         
    target_len = min(valid_lengths) # Trim everyone to the shortest VALID length
    #print(f"Trimming data to common length: {target_len} samples.")
    
    skipped_short = 0
    for sub in valid_subjects:
        data = aligned_data[sub]
        if len(data) < target_len:
            # This subject is too short (below threshold), skip
            # print(f"  Dropping {sub}: length {len(data)} < {target_len}")
            skipped_short += 1
            continue
        
        # Trim to target_len
        clean_data_dict[sub] = data[:target_len]
        final_subjects.append(sub)
        
    if skipped_short > 0:
        print(f"Dropped {skipped_short} subjects due to insufficient data length.")
        
    #print(f"Successfully loaded {len(final_subjects)} subjects for {signal_type}.\n")
    
    # Create DataFrame
    df_final = pd.DataFrame(clean_data_dict)
    
    return df_final


def sliding_window_split_half_isc(df_group, win_size, step_size, fs, n_splits=50):
    """
    Calculates Dynamic ISC using a Bootstrapped Split-Half approach.
    
    For each window, subjects are randomly divided into two halves multiple times (n_splits).
    The correlation is calculated between the Mean Signal of Half A and the Mean Signal of Half B.
    
    Args:
        df_group: DataFrame or Array of shape (Time, Subjects)
        win_size: Window size in samples
        step_size: Step size in samples
        fs: Sampling rate
        n_splits: Number of random permutations to average per window (default 50)
        
    Returns:
        DataFrame with ['Time_Sec', 'Mean_r']
    """
    data = df_group.values
    n_samples, n_subs = data.shape
    
    # Safety check
    if n_subs < 2:
        raise ValueError("Need at least 2 subjects to perform split-half ISC.")
    
    half_size = n_subs // 2
    results = []
    
    # Iterate through time with sliding window
    for start_idx in range(0, n_samples - win_size, step_size):
        end_idx = start_idx + win_size
        time_sec = end_idx / fs  # Causal timestamp
        
        # Extract window data (Time x Subjects)
        window_data = data[start_idx:end_idx, :]
        
        split_correlations = []
        
        # --- BOOTSTRAP PERMUTATION LOOP ---
        for _ in range(n_splits):
            # 1. Randomly shuffle indices
            shuffled_indices = np.random.permutation(n_subs)
            
            # 2. Split into two groups
            # If n_subs is odd, the last subject is excluded from this specific permutation
            idx_a = shuffled_indices[:half_size]
            idx_b = shuffled_indices[half_size : 2*half_size]
            
            # 3. Calculate Means of each half
            mean_a = np.mean(window_data[:, idx_a], axis=1)
            mean_b = np.mean(window_data[:, idx_b], axis=1)
            
            # 4. Calculate Correlation
            # Check for flatlines (std=0) to avoid NaNs
            if np.std(mean_a) == 0 or np.std(mean_b) == 0:
                r = np.nan
            else:
                r = np.corrcoef(mean_a, mean_b)[0, 1]
            
            split_correlations.append(r)
        
        # --- FISHER Z-AVERAGE ---
        clean_rs = [x for x in split_correlations if not np.isnan(x)]
        
        if len(clean_rs) > 0:
            # Clip to avoid infinity in arctanh
            z_vals = np.arctanh(np.clip(clean_rs, -0.999, 0.999))
            mean_r = np.tanh(np.mean(z_vals))
        else:
            mean_r = 0 
            
        results.append({'Time_Sec': time_sec, 'Mean_r': mean_r})
        
    return pd.DataFrame(results)
    
def sliding_window_within_group_isc(df_group, win_size, step_size, fs):
    """
    Calculates Dynamic ISC within a single group using Leave-One-Out (LOO) approach.
    
    Args:
        df_group: DataFrame or Array of shape (Time, Subjects)
        win_size: Window size in samples
        step_size: Step size in samples
        fs: Sampling rate (for time axis calculation)
        
    Returns:
        DataFrame with ['Time_Sec', 'Mean_r']
    """
    data = df_group.values
    n_samples, n_subs = data.shape
    
    results = []
    
    # Iterate through time with sliding window
    for start_idx in range(0, n_samples - win_size, step_size):
        end_idx = start_idx + win_size
        time_sec = end_idx / fs  # Causal timestamp (time at end of window)
        
        # Extract window data (Time x Subjects)
        window_data = data[start_idx:end_idx, :]
        
        loo_correlations = []
        
        # --- LEAVE-ONE-OUT LOOP ---
        for i in range(n_subs):
            # 1. Subject i
            subject_ts = window_data[:, i]
            
            # 2. Average of everyone else (LOO)
            others_idx = [j for j in range(n_subs) if j != i]
            group_mean_ts = np.mean(window_data[:, others_idx], axis=1)
            
            # 3. Correlation (handle potential flat-lines/zeros)
            if np.std(subject_ts) == 0 or np.std(group_mean_ts) == 0:
                r = np.nan # Undefined correlation
            else:
                r = np.corrcoef(subject_ts, group_mean_ts)[0, 1]
            
            loo_correlations.append(r)
        
        # --- FISHER Z-AVERAGE ---
        # Convert r -> z, take mean, convert z -> r
        # Filter out NaNs if any bad segments occurred
        clean_rs = [x for x in loo_correlations if not np.isnan(x)]
        
        if len(clean_rs) > 0:
            z_vals = np.arctanh(np.clip(clean_rs, -0.999, 0.999))
            mean_r = np.tanh(np.mean(z_vals))
        else:
            mean_r = 0 # or NaN
            
        results.append({'Time_Sec': time_sec, 'Mean_r': mean_r})
        
    return pd.DataFrame(results)
    


# ==========================================
# HELPER FUNCTIONS FOR ANALYSIS
# ==========================================
def compute_pairwise_matrix(df):
    """Computes NxN Correlation Matrix from a Time x Subject DataFrame."""
    # Transpose: np.corrcoef needs Rows=Variables (Subjects)
    return np.corrcoef(df.values.T)

def compute_loo_score(df):
    """
    Computes Leave-One-Out (LOO) ISC score for each subject.
    Returns: Array of shape (N_subjects,)
    """
    data = df.values.T # Subjects x Time
    n_subs, n_time = data.shape
    scores = []
    
    for i in range(n_subs):
        subj_ts = data[i, :]
        
        # Mask out the current subject to create the group average
        others_mask = np.ones(n_subs, dtype=bool)
        others_mask[i] = False
        group_ts = np.mean(data[others_mask, :], axis=0)
        
        # Handle flat signals (e.g. artifacts) to avoid NaNs
        if np.std(subj_ts) == 0 or np.std(group_ts) == 0:
            scores.append(0)
        else:
            r = np.corrcoef(subj_ts, group_ts)[0, 1]
            scores.append(r)
        
    return np.array(scores)

