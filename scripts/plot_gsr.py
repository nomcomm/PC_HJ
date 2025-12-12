import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Define paths
PROCESSED_DATA_DIR = 'data/01_processed_data/00_processed_GSR'
OUTPUT_PLOT_PATH = os.path.join(PROCESSED_DATA_DIR, 'all_subjects_gsr_plot.png')

def plot_all_subjects():
    # Find all processed CSV files
    csv_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, '*_processed_gsr.csv'))
    print(f"Found {len(csv_files)} processed files to plot.")
    
    if not csv_files:
        print("No files found. Exiting.")
        return

    plt.figure(figsize=(15, 8))
    
    for file_path in csv_files:
        try:
            filename = os.path.basename(file_path)
            subject_id = filename.split('_')[0]
            
            df = pd.read_csv(file_path)
            
            plt.plot(df['Time_sec'], df['GSR_Interpolated'], alpha=0.3, linewidth=1, label=subject_id)
            
        except Exception as e:
            print(f"Error reading/plotting {file_path}: {e}")

    plt.title("GSR Time Course - All Subjects (2Hz)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("GSR Interpolated (µS)")
    plt.grid(True, alpha=0.3)
    
    # Too many subjects for legend likely, so omitting or simplifying
    # plt.legend() 
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150)
    print(f"Plot saved to {OUTPUT_PLOT_PATH}")

if __name__ == "__main__":
    plot_all_subjects()
