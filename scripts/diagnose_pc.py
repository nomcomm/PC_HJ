import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.stats import zscore

# Define paths
PROCESSED_DATA_DIR = 'data/01_processed_data/00_processed_GSR'
CONDITIONS_FILE = 'data/sub_conditions.csv'
OUTPUT_PLOT_PATH = os.path.join(PROCESSED_DATA_DIR, 'diagnostic_plot_PC.png')

def diagnose_pc():
    # Load conditions
    try:
        conditions_df = pd.read_csv(CONDITIONS_FILE)
        conditions_df['SubjectID'] = conditions_df['RespondentName'].apply(lambda x: x.split('_')[0])
        condition_map = dict(zip(conditions_df['SubjectID'], conditions_df['SourceStimulus']))
    except Exception as e:
        print(f"Error loading conditions file: {e}")
        return

    # Find processed files
    csv_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, '*_processed_gsr.csv'))
    csv_files.sort()

    pc_data = []
    pc_ids = []
    time_points = None

    print("Loading PC data...")
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        subject_id = filename.split('_')[0]
        
        condition = condition_map.get(subject_id)
        if condition != 'PC':
            continue

        try:
            df = pd.read_csv(file_path)
            if time_points is None:
                time_points = df['Time_sec'].values
            
            gsr_trace = df['GSR_Interpolated'].values
            if np.std(gsr_trace) == 0:
                 continue
            
            gsr_z = zscore(gsr_trace, nan_policy='omit')
            pc_data.append(gsr_z)
            pc_ids.append(subject_id)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Convert to matrix
    data_matrix = np.array(pc_data)
    n_subjects = len(pc_ids)
    print(f"Loaded {n_subjects} PC subjects.")

    # Split halves
    mid_point = n_subjects // 2
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Plot First Half (faint blue)
    for i in range(mid_point):
        plt.plot(time_points, data_matrix[i, :], color='blue', alpha=0.05)
        
    # Plot Second Half (faint orange) - These are the ones interfering?
    for i in range(mid_point, n_subjects):
        # We can make these slightly more visible or color by index to identify specific offenders
        plt.plot(time_points, data_matrix[i, :], color='orange', alpha=0.1)

    # Plot Means
    mean_1 = np.nanmean(data_matrix[:mid_point, :], axis=0)
    mean_2 = np.nanmean(data_matrix[mid_point:, :], axis=0)
    
    plt.plot(time_points, mean_1, color='blue', linewidth=3, label='First Half Mean')
    plt.plot(time_points, mean_2, color='orange', linewidth=3, label='Second Half Mean')
    
    plt.title("Diagnostic Plot: PC Condition Individual Traces (Z-Scored)")
    plt.xlabel("Time (s)")
    plt.ylabel("GSR (Z)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150)
    print(f"Diagnostic plot saved to {OUTPUT_PLOT_PATH}")

if __name__ == "__main__":
    diagnose_pc()
