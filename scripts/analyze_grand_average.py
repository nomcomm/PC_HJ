import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.stats import zscore

# Define paths
PROCESSED_DATA_DIR = 'data/01_processed_data/00_processed_GSR'
OUTPUT_PLOT_PATH = os.path.join(PROCESSED_DATA_DIR, 'grand_average_split_half_{condition}.png')
CONDITIONS_FILE = 'data/sub_conditions.csv'

def analyze_grand_average():
    # Load conditions
    try:
        conditions_df = pd.read_csv(CONDITIONS_FILE)
        # Create a dictionary mapping subject ID (e.g., 'sub001') to condition (e.g., 'PCM')
        # We need to extract the ID from 'RespondentName'. 
        # RespondentName example: 'sub001_11-11-25 13h40m' -> 'sub001'
        conditions_df['SubjectID'] = conditions_df['RespondentName'].apply(lambda x: x.split('_')[0])
        condition_map = dict(zip(conditions_df['SubjectID'], conditions_df['SourceStimulus']))
        print(f"Loaded conditions for {len(condition_map)} subjects.")
    except Exception as e:
        print(f"Error loading conditions file: {e}")
        return

    # Find all processed CSV files
    csv_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, '*_processed_gsr.csv'))
    print(f"Found {len(csv_files)} processed files.")
    
    if not csv_files:
        print("No files found. Exiting.")
        return

    # Sort files by subject ID
    csv_files.sort()

    # Data structure to hold data by condition
    # data_by_condition = {'PC': [], 'PCM': []}
    data_by_condition = {}

    print("Loading and z-scoring data...")
    for file_path in csv_files:
        try:
            filename = os.path.basename(file_path)
            subject_id = filename.split('_')[0]
            
            # Lookup condition
            condition = condition_map.get(subject_id)
            if not condition:
                print(f"Warning: No condition found for {subject_id}. Skipping.")
                continue
            
            if condition not in data_by_condition:
                data_by_condition[condition] = []

            df = pd.read_csv(file_path)
            
            # Note: We assume time points are consistent now (N=638) thanks to process_gsr.py fix
            time_points = df['Time_sec'].values
            gsr_trace = df['GSR_Interpolated'].values
            
            if np.std(gsr_trace) == 0:
                 print(f"Warning: {subject_id} has constant GSR (std=0). Skipping z-score/inclusion.")
                 continue
                 
            gsr_z = zscore(gsr_trace, nan_policy='omit')
            
            data_by_condition[condition].append(gsr_z)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Process each condition
    for condition, data_list in data_by_condition.items():
        print(f"\nAnalyzing Condition: {condition}")
        
        data_matrix = np.array(data_list)
        n_subjects = data_matrix.shape[0]
        print(f"  Included {n_subjects} subjects.")
        
        if n_subjects < 2:
            print("  Not enough subjects for split-half. Skipping.")
            continue

        # Split into two halves (first half vs second half of sorted list)
        mid_point = n_subjects // 2
        first_half_data = data_matrix[:mid_point, :]
        second_half_data = data_matrix[mid_point:, :]
        
        print(f"  Group 1: {mid_point}, Group 2: {n_subjects - mid_point}")

        # Calculate Grand Averages
        mean_1 = np.nanmean(first_half_data, axis=0)
        mean_2 = np.nanmean(second_half_data, axis=0)
        
        ste_1 = np.nanstd(first_half_data, axis=0) / np.sqrt(first_half_data.shape[0])
        ste_2 = np.nanstd(second_half_data, axis=0) / np.sqrt(second_half_data.shape[0])

        # Plotting
        plt.figure(figsize=(15, 8))
        
        plt.plot(time_points, mean_1, label=f'First Half (n={mid_point})', color='blue', linewidth=2)
        plt.fill_between(time_points, mean_1 - ste_1, mean_1 + ste_1, color='blue', alpha=0.1)
        
        plt.plot(time_points, mean_2, label=f'Second Half (n={n_subjects - mid_point})', color='orange', linewidth=2)
        plt.fill_between(time_points, mean_2 - ste_2, mean_2 + ste_2, color='orange', alpha=0.1)

        valid_mask = ~np.isnan(mean_1) & ~np.isnan(mean_2)
        correlation = np.corrcoef(mean_1[valid_mask], mean_2[valid_mask])[0, 1]
        
        plt.title(f"Condition {condition}: Split-Half Grand Average GSR (Z-Scored) - r={correlation:.2f}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("GSR (Z-Score)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = OUTPUT_PLOT_PATH.format(condition=condition)
        plt.savefig(plot_path, dpi=150)
        print(f"  Plot saved to {plot_path}")
        plt.close() # Close figure to free memory

if __name__ == "__main__":
    analyze_grand_average()
