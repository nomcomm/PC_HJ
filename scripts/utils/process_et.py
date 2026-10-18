import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
from pathlib import Path

# --- Configuration ---

# --- DYNAMIC PATH SETUP ---
# 1. Resolve the path to THIS file (main_path/scripts/utils/gsr_process.py)
script_location = Path(__file__).resolve()

# 2. Calculate Root (walk up 3 levels: utils -> scripts -> main_path)
PROJECT_ROOT = script_location.parent.parent.parent

# 3. Define Data Paths relative to Root
RAW_DATA_DIR = PROJECT_ROOT / 'data' / '00_raw_data' / '02_ET_data'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / '01_processed_data' / '02_processed_ET'

# Convert to string (for safety with legacy libs)
RAW_DATA_DIR = str(RAW_DATA_DIR)
PROCESSED_DATA_DIR = str(PROCESSED_DATA_DIR)


STIMULUS_DURATION_SEC = 319
# We will detect sampling rate dynamically, but expected is roughly 60Hz. 
# If 60Hz, 319s * 60 = 19140 samples.

def process_subject(file_path):
    """
    Processes a single subject's ET data file.
    """
    try:
        # Read the file to finding the header row
        # The header seems to start at a line starting with "Row,Timestamp" or similar.
        # Based on inspection, it's roughly line 28/29. Let's find it dynamically.
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        header_row_idx = -1
        for i, line in enumerate(lines):
            if line.startswith("Row,Timestamp"):
                header_row_idx = i
                break
        
        if header_row_idx == -1:
            print(f"Error: Could not find header row in {file_path}")
            return None

        # Read CSV skipping metadata lines
        df = pd.read_csv(file_path, skiprows=header_row_idx)
        
        # --- Identify Stimulus Window ---
        # Look for SlideEvent == "StartSlide" (or StartMedia) and StimType == "TestImage" (or similar unique identifier for the movie)
        # In the inspected file: 
        # Row 29: SlideEvent=StartSlide, StimType=TestImage, SourceStimuliName=PCM
        # Row 32: SlideEvent=StartMedia, ...
        # We need to find the START of the relevant stimulus.
        
        # Filter for rows where stimulus starts
        start_events = df[
            (df["SlideEvent"].isin(["StartSlide", "StartMedia"])) & 
            (df["StimType"] == "TestImage") # Assuming 'TestImage' is the consistent tag for the movie
        ]
        
        if start_events.empty:
             # Fallback: try finding just the first StartMedia event if TestImage isn't found
             start_events = df[df["SlideEvent"] == "StartMedia"]
             if start_events.empty:
                print(f"Warning: No start event found for {os.path.basename(file_path)}")
                return None

        # Take the first occurrence as the start (assuming only one movie presentation)
        start_idx = start_events.index[0]
        start_timestamp = df.loc[start_idx, "Timestamp"]
        
        # Estimate sampling rate
        # Calculate mean time difference between consecutive samples in a small window
        time_diffs = df["Timestamp"].diff().dropna().head(1000)
        mean_dt_ms = time_diffs.mean() # Timestamp is usually in ms
        if mean_dt_ms == 0 or np.isnan(mean_dt_ms):
             print(f"Warning: Could not determine sampling rate for {os.path.basename(file_path)}")
             return None
             
        sampling_rate = 1000 / mean_dt_ms
        expected_samples = int(STIMULUS_DURATION_SEC * sampling_rate)
        
        # print(f"File: {os.path.basename(file_path)}, Rate: {sampling_rate:.2f}Hz, Expected Samples: {expected_samples}")

        # Extract data window
        # We want data FROM start_idx
        # We need to ensure we have enough data.
        end_idx = start_idx + expected_samples
        
        if end_idx > len(df):
            print(f"Warning: Not enough data for {os.path.basename(file_path)}. Needed {expected_samples}, have {len(df)-start_idx}")
            # Identify if it's just a few samples short or major data loss. For now, crop to what we have?
            # User wants chopping out 3m 19s. If short, we might pad or just cut short.
            # Let's cut to min(end_idx, len(df))
            end_idx = len(df)

        stimulus_data = df.iloc[start_idx:end_idx].copy()
        
        # --- Processing Pupil Data ---
        # Replace -1 with NaN
        stimulus_data["ET_PupilLeft"] = stimulus_data["ET_PupilLeft"].replace(-1, np.nan)
        stimulus_data["ET_PupilRight"] = stimulus_data["ET_PupilRight"].replace(-1, np.nan)
        
        # Interpolate missing values
        # Limit interpolation? e.g., limit=something (ms). User said "deal with missing data (interpolation)". 
        # Standard practice is linear interpolation.
        stimulus_data["ET_PupilLeft"] = stimulus_data["ET_PupilLeft"].interpolate(method='linear', limit_direction='both')
        stimulus_data["ET_PupilRight"] = stimulus_data["ET_PupilRight"].interpolate(method='linear', limit_direction='both')
        
        # Harmonize (Average)
        # If one is NaN, use the other. If both valid, average.
        stimulus_data["Mean_Pupil"] = stimulus_data[["ET_PupilLeft", "ET_PupilRight"]].mean(axis=1)
        
        # --- Processing Gaze Data ---
        # Columns: 'Interpolated Gaze X', 'Interpolated Gaze Y' (Wait, in CSV headers it was 'Interpolated Gaze X')
        # Let's verify column names.
        # In the file viewed: "Interpolated Gaze X", "Interpolated Gaze Y"
        # And "Gaze X", "Gaze Y". User asked for Interpolated ones.
        
        gaze_cols = ["Interpolated Gaze X", "Interpolated Gaze Y"]
        # Check if columns exist
        if not all(col in stimulus_data.columns for col in gaze_cols):
             # Try finding them without spaces or similar? The view_file output showed "Interpolated Gaze X"
             pass

        for col in gaze_cols:
            if col in stimulus_data.columns:
                stimulus_data[col] = stimulus_data[col].replace(-1, np.nan) # Assuming -1 is invalid for Gaze too
                stimulus_data[col] = stimulus_data[col].interpolate(method='linear', limit_direction='both')
        
        # Prepare output DataFrame
        output_df = stimulus_data[["Timestamp", "ET_PupilLeft", "ET_PupilRight", "Mean_Pupil"] + gaze_cols].copy()
        
        # Reset index
        output_df.reset_index(drop=True, inplace=True)
        
        return output_df

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None

def main():
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        
    raw_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    print(f"Found {len(raw_files)} raw ET files.")
    
    for file_path in tqdm(raw_files):
        processed_df = process_subject(file_path)
        
        if processed_df is not None:
            # Generate output filename
            base_name = os.path.basename(file_path)
            # Remove date/time clutter if desired, or keep unique. 
            # sub001_<session date and time>.csv -> sub001_processed_et.csv
            # Extract subject ID (first part split by _)
            sub_id = base_name.split('_')[0]
            output_filename = f"{sub_id}_processed_et.csv"
            output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
            
            processed_df.to_csv(output_path, index=False)

    print("Processing complete.")

if __name__ == "__main__":
    main()
