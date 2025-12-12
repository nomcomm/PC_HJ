import pandas as pd
import numpy as np
import os
import glob
from scipy import signal

# Define paths
# Define paths
RAW_DATA_DIR = 'data/00_raw_data/01_ECG_data'
PROCESSED_DATA_DIR = 'data/01_processed_data/01_processed_ECG'

# Ensure output directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def process_subject(file_path):
    filename = os.path.basename(file_path)
    subject_id = filename.split('_')[0]
    print(f"Processing {subject_id}...")

    try:
        # Dynamically find the header line
        header_row = 0
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('Row,Timestamp'):
                    header_row = i
                    break
        
        # Read CSV with detected header row
        # Note: read_csv 'header' argument is 0-indexed relative to the file start if we don't use skiprows
        # If we use header=header_row, it skips lines before it automatically
        df = pd.read_csv(file_path, header=header_row)
        
        # Check if columns exist
        # We target 'Heart Rate ECG LL-RA ALG' or similar based on file inspection
        target_col = 'Heart Rate ECG LL-RA ALG'
        
        if target_col not in df.columns:
             # Fallback check
             if 'Heart Rate' in df.columns:
                 target_col = 'Heart Rate'
             else:
                 # Check for fuzzy match
                 hr_cols = [c for c in df.columns if 'Heart Rate' in c]
                 if hr_cols:
                     target_col = hr_cols[0]
                 else:
                     print(f"  WARNING: 'Heart Rate' column not found in {filename}. Skipping.")
                     return

        # Filter relevant columns
        required_cols = ['Timestamp', 'SlideEvent', target_col]
        # Ensure SlideEvent exists too
        if 'SlideEvent' not in df.columns:
             print(f"  WARNING: 'SlideEvent' column not found in {filename}. Skipping.")
             return

        df_subset = df[['Timestamp', 'SlideEvent', target_col]].copy()
        
        # Find StartMedia and EndMedia events
        start_indices = df_subset.index[df_subset['SlideEvent'] == 'StartMedia'].tolist()
        end_indices = df_subset.index[df_subset['SlideEvent'] == 'EndMedia'].tolist()

        if not start_indices or not end_indices:
            print(f"  WARNING: StartMedia or EndMedia event not found in {filename}. Skipping.")
            return

        # Assuming the first StartMedia and the (expected matching) EndMedia
        start_idx = start_indices[0]
        # Find the first EndMedia after start_idx
        valid_end_indices = [i for i in end_indices if i > start_idx]
        if not valid_end_indices:
             print(f"  WARNING: No EndMedia found after StartMedia in {filename}. Skipping.")
             return
        end_idx = valid_end_indices[0]

        # Cut the data
        # We include the start and end points
        df_cut = df_subset.iloc[start_idx:end_idx+1].copy()
        
        # Calculate duration to verify
        duration_ms = df_cut['Timestamp'].iloc[-1] - df_cut['Timestamp'].iloc[0]
        duration_sec = duration_ms / 1000.0
        print(f"  Duration: {duration_sec:.2f} seconds")

        # Resample to 2Hz
        # Force fixed number of samples to N=638 (based on 319s * 2Hz)
        target_fs = 2.0
        EXPECTED_SAMPLES = 638
        
        calculated_samples = int(duration_sec * target_fs)
        
        if abs(calculated_samples - EXPECTED_SAMPLES) > 10:
             print(f"  WARNING: Large duration mismatch. Calculated samples: {calculated_samples}, Expected: {EXPECTED_SAMPLES}. Skipping to be safe.")
             return

        num_samples = EXPECTED_SAMPLES
        
        original_signal = df_cut[target_col].values
        
        # Handle potential non-numeric data
        original_signal = pd.to_numeric(original_signal, errors='coerce')
        
        # Treat -1 as NaN (invalid sensor data)
        original_signal[original_signal == -1] = np.nan
        
        # Interpolate NaNs
        if np.isnan(original_signal).any():
             nans, x= np.isnan(original_signal), lambda z: z.nonzero()[0]
             # Handle edge case where all are NaN
             if nans.all():
                 print(f"  WARNING: All Heart Rate values are NaN (or -1) for {filename}. Skipping.")
                 return
             original_signal[nans]= np.interp(x(nans), x(~nans), original_signal[~nans])
             
        print(f"  Mean Heart Rate before resampling: {np.mean(original_signal):.2f}")

        # NO High-pass filtering for ECG/Heart Rate as requested
        
        # Resample
        resampled_signal = signal.resample(original_signal, num_samples)
        
        # Create new time array for resampled data
        new_time = np.linspace(0, 319.0, num_samples)
        
        # Create result dataframe
        df_resampled = pd.DataFrame({
            'Time_sec': new_time,
            'Heart_Rate': resampled_signal
        })
        
        # Save to CSV
        output_filename = f"{subject_id}_processed_ecg.csv"
        output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
        df_resampled.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")

    except Exception as e:
        print(f"  ERROR processing {filename}: {e}")

def main():
    # Find all CSV files
    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, '*.csv'))
    print(f"Found {len(csv_files)} files.")
    
    for file_path in csv_files:
        process_subject(file_path)

if __name__ == "__main__":
    main()
