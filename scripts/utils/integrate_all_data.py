
import pandas as pd
import numpy as np
import os
import glob
from scipy.interpolate import interp1d


# Paths
# Determine directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/01_processed_data")
GSR_DIR = os.path.join(DATA_DIR, "00_processed_GSR")
ECG_DIR = os.path.join(DATA_DIR, "01_processed_ECG")
ET_DIR = os.path.join(DATA_DIR, "02_processed_ET")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/02_processed_combined_data")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
TARGET_FS = 20.0  # Hz
DT = 1.0 / TARGET_FS
DURATION = 319.0  # Seconds (matches observed data duration)
TIME_AXIS = np.arange(0, DURATION + DT/1000, DT) # + DT/1000 to include the last point safely

def process_subject(subject_id):
    """
    Integrates data for a single subject.
    """
    print(f"Processing {subject_id}...")
    
    # 1. File Paths
    gsr_path = os.path.join(GSR_DIR, f"{subject_id}_processed_gsr.csv")
    ecg_path = os.path.join(ECG_DIR, f"{subject_id}_processed_ecg.csv")
    et_path = os.path.join(ET_DIR, f"{subject_id}_processed_et.csv")
    
    # 2. Check existence
    if not (os.path.exists(gsr_path) and os.path.exists(ecg_path) and os.path.exists(et_path)):
        print(f"  -> Missing files for {subject_id}. Skipping.")
        if not os.path.exists(gsr_path): print(f"     Missing: {gsr_path}")
        if not os.path.exists(ecg_path): print(f"     Missing: {ecg_path}")
        if not os.path.exists(et_path): print(f"     Missing: {et_path}")
        return

    # 3. Load Data
    try:
        df_gsr = pd.read_csv(gsr_path)
        df_ecg = pd.read_csv(ecg_path)
        df_et = pd.read_csv(et_path)
    except Exception as e:
        print(f"  -> Error loading files for {subject_id}: {e}")
        return

    # 4. Interpolate GSR
    # GSR/ECG expected format: Time_sec, Value
    # Interpolate to common TIME_AXIS
    # Use 'nearest' or 'linear' extrapolation? Assuming data covers 0-319 roughly.
    # GSR files seem to cover 0 to 319 exactly based on previous checks.
    
    # --- GSR ---
    # Handle potential duplicate times or NaNs if any (should be clean from prev steps but good to be safe)
    if df_gsr['Time_sec'].duplicated().any():
        df_gsr = df_gsr.drop_duplicates(subset='Time_sec')
    
    f_gsr = interp1d(df_gsr['Time_sec'], df_gsr.iloc[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
    gsr_interp = f_gsr(TIME_AXIS)

    # --- Heart Rate ---
    if df_ecg['Time_sec'].duplicated().any():
        df_ecg = df_ecg.drop_duplicates(subset='Time_sec')
        
    f_hr = interp1d(df_ecg['Time_sec'], df_ecg.iloc[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
    hr_interp = f_hr(TIME_AXIS)

    # --- Eye Tracking ---
    # ET format: Timestamp, ... Mean_Pupil, Interpolated Gaze X, Interpolated Gaze Y
    # Align ET Time: t = (Timestamp - T0) / 1000
    if len(df_et) == 0:
        print(f"  -> Empty ET file for {subject_id}. Skipping.")
        return

    t0_et = df_et['Timestamp'].iloc[0]
    et_time_sec = (df_et['Timestamp'] - t0_et) / 1000.0
    
    # Handle NaNs in source ET data (though prev steps might have interpolated, check columns)
    # Target columns: Mean_Pupil, Interpolated Gaze X, Interpolated Gaze Y
    # Note: Column names might vary slightly? Based on file check: "Interpolated Gaze X"
    
    pupil_src = df_et['Mean_Pupil']
    gaze_x_src = df_et['Interpolated Gaze X']
    gaze_y_src = df_et['Interpolated Gaze Y']

    # Interpolate ET
    # Use linear interpolation. For gaps, the previous processing filled them?
    # If there are NaNs in 'processed' ET, we can leave them as NaNs or fill.
    # interp1d doesn't handle NaNs well in x or y. 
    # Let's assume processed ET is clean-ish. If NaNs persist, we might get NaNs in output.
    
    def safe_interp(time_src, data_src, time_target):
        # Drop NaNs from source for interpolation definition
        mask = ~np.isnan(data_src)
        if mask.sum() < 2:
            return np.full_like(time_target, np.nan)
        
        f = interp1d(time_src[mask], data_src[mask], kind='linear', bounds_error=False, fill_value="extrapolate")
        return f(time_target)

    pupil_interp = safe_interp(et_time_sec, pupil_src, TIME_AXIS)
    gaze_x_interp = safe_interp(et_time_sec, gaze_x_src, TIME_AXIS)
    gaze_y_interp = safe_interp(et_time_sec, gaze_y_src, TIME_AXIS)

    # 5. Combine DataFrame
    df_combined = pd.DataFrame({
        'Time_sec': TIME_AXIS,
        'GSR': gsr_interp,
        'Heart_Rate': hr_interp,
        'Pupil_Mean': pupil_interp,
        'Gaze_X': gaze_x_interp,
        'Gaze_Y': gaze_y_interp
    })

    # 6. Save
    output_path = os.path.join(OUTPUT_DIR, f"{subject_id}_combined_physio.csv")
    df_combined.to_csv(output_path, index=False)
    # print(f"  -> Saved to {output_path}")

def main():
    # Find all subjects from GSR folder
    gsr_files = glob.glob(os.path.join(GSR_DIR, "*_processed_gsr.csv"))
    subjects = [os.path.basename(f).split('_')[0] for f in gsr_files]
    subjects.sort()
    
    print(f"Found {len(subjects)} subjects to process.")
    
    for sub in subjects:
        process_subject(sub)

    print("Done!")

if __name__ == "__main__":
    main()
