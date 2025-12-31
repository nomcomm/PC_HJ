
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os, sys, re
from scipy.stats import pearsonr
from pathlib import Path
import glob



def plot_time_series(data_df, title="Group Time Series", output_path=None):
    """
    Plots Grand Average with confidence intervals and/or individual faint traces.
    """
    plt.figure(figsize=(15, 6))
    
    # Convert to array for easier handling
    data = data_df.values
    time_len = data.shape[0]
    time_axis = np.arange(time_len) # or actual time if passed
    
    # 1. Individual traces (faint)
    plt.plot(time_axis, data, color='gray', alpha=0.03, linewidth=0.8)
    
    # 2. Grand Average
    grand_avg = np.mean(data, axis=1)
    std_err = np.std(data, axis=1) / np.sqrt(data.shape[1])
    
    plt.plot(time_axis, grand_avg, color='black', linewidth=2, label='Grand Average')
    plt.fill_between(time_axis, grand_avg - 1.96*std_err, grand_avg + 1.96*std_err, 
                     color='black', alpha=0.2, label='95% CI')
    
    plt.title(title)
    plt.xlabel("Time (samples)")
    plt.ylabel("Signal Amplitude (Z-scored)")
    plt.legend(loc='upper right')
    plt.xlim(0, time_len)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Time series plot saved to {output_path}")
    plt.close()

def plot_split_half_comparison(data_df, title="Split-Half Consistency", output_path=None):
    """
    Visualizes two random halves of the group to show consistency.
    """
    data = data_df.values
    n_subjects = data.shape[1]
    indices = np.arange(n_subjects)
    np.random.shuffle(indices)
    mid = n_subjects // 2
    
    g1 = data[:, indices[:mid]].mean(axis=1)
    g2 = data[:, indices[mid:]].mean(axis=1)
    
    plt.figure(figsize=(10, 4))
    plt.plot(g1, color='dodgerblue', alpha=0.9, linewidth=1.5, label='Split Group 1')
    plt.plot(g2, color='darkorange', alpha=0.9, linewidth=1.5, label='Split Group 2')
    
    # Correlation between them
    r = np.corrcoef(g1, g2)[0, 1]
    
    plt.title(f"{title} (r = {r:.3f})")
    plt.xlabel("Time (samples)")
    plt.ylabel("Signal Amplitude")
    plt.legend()
    plt.xlim(0, len(g1))
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Split-half plot saved to {output_path}")
    plt.show()
    #plt.close()

def plot_isc_distributions(pairwise_data, loo_data, title="ISC Distributions", output_path=None):
    """
    Plots densities of Pairwise and LOO ISC values.
    """
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(pairwise_data, fill=True, color='teal', label=f'Pairwise (Mean={np.mean(pairwise_data):.3f})')
    sns.kdeplot(loo_data, fill=True, color='purple', label=f'Leave-One-Out (Mean={np.mean(loo_data):.3f})')
    
    plt.title(title)
    plt.xlabel("Correlation (r)")
    plt.ylabel("Density")
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Distribution plot saved to {output_path}")
    plt.close()

def plot_correlation_matrix(matrix, subject_ids=None, title="Pairwise Correlation Matrix", output_path=None):
    """
    Heatmap of the pairwise correlation matrix.
    """
    plt.figure(figsize=(12, 10))
    
    if subject_ids is None:
        xticklabels = False
        yticklabels = False
    else:
        # If too many subjects, skip labels
        if len(subject_ids) > 50:
            xticklabels = False
            yticklabels = False
        else:
            xticklabels = subject_ids
            yticklabels = subject_ids
            
    sns.heatmap(matrix, center=0, cmap='RdBu_r', vmin=-1, vmax=1, 
                xticklabels=xticklabels, yticklabels=yticklabels, square=True)
    
    plt.title(title)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Matrix plot saved to {output_path}")
    plt.close()


def get_seconds_to_indices(start_sec, end_sec, total_samples=638, total_duration=319):
    """Converts seconds to sample indices."""
    start_idx = int((start_sec / total_duration) * total_samples)
    end_idx = int((end_sec / total_duration) * total_samples)
    return start_idx, end_idx

def extract_segments(data_df, condition_segments, total_samples=638):
    """
    Slices the dataframe into a dictionary of segments {seg_id: slice_df}.
    """
    seg_data = {}
    n_samples = data_df.shape[0]
    
    for seg_id, (start_s, end_s) in condition_segments.items():
        s_idx, e_idx = get_seconds_to_indices(start_s, end_s, n_samples, 319)
        # Ensure bounds
        s_idx = max(0, s_idx)
        e_idx = min(n_samples, e_idx)
        
        seg_data[seg_id] = data_df.iloc[s_idx:e_idx]
        
    return seg_data



def compute_pairwise_isc(data, return_matrix=False):
    """
    Computes pairwise Inter-Subject Correlation (ISC).
    
    Args:
        data (pd.DataFrame or np.ndarray): Time x Subjects data.
        return_matrix (bool): If True, returns the full correlation matrix.
        
    Returns:
        mean_r (float): Average pairwise correlation (upper triangle).
        pairwise_corrs (np.ndarray): Array of all pairwise correlation coefficients.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
        
    n_samples, n_subjects = data.shape
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(data.T)
    
    # Extract upper triangle indices (k=1 excludes diagonal)
    triu_indices = np.triu_indices(n_subjects, k=1)
    pairwise_corrs = corr_matrix[triu_indices]
    
    mean_r = np.mean(pairwise_corrs)
    
    if return_matrix:
        return mean_r, pairwise_corrs, corr_matrix
    return mean_r, pairwise_corrs

def compute_loo_isc(data):
    """
    Computes Leave-One-Out (Subject-to-Group) ISC.
    Correlates each subject with the average of all other subjects.
    
    Args:
        data (pd.DataFrame or np.ndarray): Time x Subjects data.
        
    Returns:
        loo_corrs (np.ndarray): Correlation for each subject.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
        
    n_samples, n_subjects = data.shape
    loo_corrs = np.zeros(n_subjects)
    
    for i in range(n_subjects):
        # Current subject
        subj_ts = data[:, i]
        
        # Average of others
        others_mask = np.ones(n_subjects, dtype=bool)
        others_mask[i] = False
        others_avg = np.mean(data[:, others_mask], axis=1)
        
        # Correlation
        r, _ = pearsonr(subj_ts, others_avg)
        loo_corrs[i] = r
        
    return loo_corrs

def compute_split_half_isc(data, n_boot=1000):
    """
    Computes Split-Half ISC using bootstrapping.
    Randomly splits group into two halves and correlates their means.
    Adjusted using Spearman-Brown prediction formula? 
    - Standard practice is often to report the raw correlation between halves as a measure of reliability,
      or apply Spearman-Brown to estimate reliability of the FULL group.
    - We will return the raw split-half correlations distribution.
    
    Args:
        data (pd.DataFrame or np.ndarray): Time x Subjects data.
        n_boot (int): Number of bootstrap iterations.
        
    Returns:
        mean_reproducibility (float): Mean correlation between random halves.
        split_corrs (np.ndarray): Distribution of correlations.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
        
    n_samples, n_subjects = data.shape
    split_corrs = np.zeros(n_boot)
    
    # Indices
    indices = np.arange(n_subjects)
    
    for b in range(n_boot):
        np.random.shuffle(indices)
        mid = n_subjects // 2
        
        group1_idx = indices[:mid]
        group2_idx = indices[mid:]
        
        # Compute means
        mean1 = np.mean(data[:, group1_idx], axis=1)
        mean2 = np.mean(data[:, group2_idx], axis=1)
        
        # Correlate
        r, _ = pearsonr(mean1, mean2)
        split_corrs[b] = r
        
    # Optional: Spearman-Brown correction to estimate full group reliability
    # r_full = (2 * r_half) / (1 + r_half)
    # We will just return the raw distribution for now as it reflects "replicability"
    
    return np.mean(split_corrs), split_corrs


def parse_subject_id(respondent_name):
    """
    Extracts subject ID (e.g., 'sub001') from RespondentName string.
    Handles potential extra spaces or underscores.
    """
    # Split by underscore and take the first part
    parts = respondent_name.split('_')
    sub_id = parts[0].strip()
    return sub_id

def get_subjects_by_condition(conditions_file, target_condition='PC'):
    """
    Reads the conditions file and returns a list of subject IDs for the target condition.
    """
    if not os.path.exists(conditions_file):
        raise FileNotFoundError(f"Conditions file not found: {conditions_file}")
        
    df = pd.read_csv(conditions_file)
    
    # Clean column names just in case
    df.columns = [c.strip() for c in df.columns]
    
    if 'RespondentName' not in df.columns or 'SourceStimulus' not in df.columns:
        raise ValueError("Conditions file must contain 'RespondentName' and 'SourceStimulus' columns.")
    
    # Filter by condition
    condition_df = df[df['SourceStimulus'].str.strip() == target_condition].copy()
    
    subject_ids = condition_df['RespondentName'].apply(parse_subject_id).tolist()
    
    # Remove duplicates if any
    subject_ids = sorted(list(set(subject_ids)))
    
    print(f"[{target_condition}] Found {len(subject_ids)} subjects in condition file.")
    return subject_ids

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
            print(f"  Warning: No data file found for subject {sub_id}")
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
                print(f"  Warning: {sub_id} {target_col} contains NaNs after interpolation. Dropping.")
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
    print(f"\nData Length Distribution for {signal_type}:")
    print(length_counts)
    
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
    print(f"Trimming data to common length: {target_len} samples.")
    
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
        
    print(f"Successfully loaded {len(final_subjects)} subjects for {signal_type}.\n")
    
    # Create DataFrame
    df_final = pd.DataFrame(clean_data_dict)
    
    return df_final




