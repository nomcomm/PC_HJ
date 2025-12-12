import pandas as pd
import numpy as np
import os
import glob
from scipy import signal

# Define paths
RAW_DATA_DIR = 'data/00_raw_data/00_GSR_data'
PROCESSED_DATA_DIR = 'data/01_processed_data/00_processed_GSR'

# Ensure output directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def process_subject(file_path):
    filename = os.path.basename(file_path)
    subject_id = filename.split('_')[0]
    print(f"Processing {subject_id}...")

    try:
        # Read CSV, skipping metadata lines (first 31 lines based on inspection)
        # We need to make sure we catch the right header line.
        # Based on inspection: line 32 starts with "Row,Timestamp,..."
        # But pandas read_csv 'header' argument is 0-indexed.
        # If line 32 is header, we skip 31 rows.
        df = pd.read_csv(file_path, skiprows=31)
        
        # Check if columns exist
        required_cols = ['Timestamp', 'SlideEvent', 'GSR Interpolated']
        # Note: 'SlideEvent' might be in a column named 'SlideEvent' or similar in the header
        # The provided file view showed: Row,Timestamp,EventSource,SlideEvent,StimType,...
        # and also 'GSR Interpolated' at column index ~14.
        
        # Let's verify column names from a potential read
        # In the file view: 
        # Line 32: Row,Timestamp,EventSource,SlideEvent,StimType,Duration,CollectionPhase,SourceStimuliName,EventSource,Preprocessed GSR,EventSource,Preprocessed GSR,EventSource,GSR Raw,GSR Interpolated,...
        
        if 'GSR Interpolated' not in df.columns:
             # Sometimes duplicate names exist, pandas might rename them like "GSR Interpolated.1"
             # Let's look for any column containing "GSR Interpolated"
             gsr_cols = [c for c in df.columns if 'GSR Interpolated' in c]
             if gsr_cols:
                 gsr_col = gsr_cols[0] # Take the first one found
             else:
                 print(f"  WARNING: 'GSR Interpolated' column not found in {filename}. Skipping.")
                 return
        else:
            gsr_col = 'GSR Interpolated'

        # Filter relevant columns
        # We need Timestamp, SlideEvent, and the GSR data
        df_subset = df[['Timestamp', 'SlideEvent', gsr_col]].copy()
        
        # Find StartMedia and EndMedia events
        start_indices = df_subset.index[df_subset['SlideEvent'] == 'StartMedia'].tolist()
        end_indices = df_subset.index[df_subset['SlideEvent'] == 'EndMedia'].tolist()

        if not start_indices or not end_indices:
            print(f"  WARNING: StartMedia or EndMedia event not found in {filename}. Skipping.")
            return

        # Assuming the first StartMedia and the (expected matching) EndMedia
        # The user mentioned roughly 4 min 19 sec duration.
        
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
        
        # We calculate the number of samples we WOULD get
        calculated_samples = int(duration_sec * target_fs)
        
        # If the difference is small (e.g., < 5 samples ~ 2.5s), we force it to EXPECTED_SAMPLES
        # This handles the slight jitter in start/end tags.
        if abs(calculated_samples - EXPECTED_SAMPLES) > 10:
             print(f"  WARNING: Large duration mismatch. Calculated samples: {calculated_samples}, Expected: {EXPECTED_SAMPLES}. Skipping to be safe.")
             return

        num_samples = EXPECTED_SAMPLES
        
        original_gsr = df_cut[gsr_col].values
        
        # Handle potential non-numeric data
        original_gsr = pd.to_numeric(original_gsr, errors='coerce')
        if np.isnan(original_gsr).any():
             nans, x= np.isnan(original_gsr), lambda z: z.nonzero()[0]
             original_gsr[nans]= np.interp(x(nans), x(~nans), original_gsr[~nans])

        # Feature: High-pass filtering to remove tonic drift
        # Order 2 Butterworth, 0.05 Hz cutoff
        # Sampling rate is ~128 Hz (we determined earlier)
        # We should filter BEFORE resampling to avoid aliasing issues? Or after?
        # Usually better before downsampling.
        # But we need sampling rate.
        # Let's assume 128 Hz or calculate from duration/length
        actual_fs = len(original_gsr) / duration_sec
        
        sos = signal.butter(2, 0.05, 'hp', fs=actual_fs, output='sos')
        filtered_gsr = signal.sosfiltfilt(sos, original_gsr)
        
        # Use filtered data for resampling
        resampled_gsr = signal.resample(filtered_gsr, num_samples)
        
        # Create new time array for resampled data
        # timestamps start at 0 for convenience in processed file? Or keep relative?
        # User asked for "cut/time-aligned" data. 
        # Usually aligning start to 0 is good for ISC.
        # Fixed duration of 319 seconds corresponding to 638 samples
        new_time = np.linspace(0, 319.0, num_samples)
        
        # Create result dataframe
        df_resampled = pd.DataFrame({
            'Time_sec': new_time,
            'GSR_Interpolated': resampled_gsr
        })
        
        # Save to CSV
        output_filename = f"{subject_id}_processed_gsr.csv"
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
