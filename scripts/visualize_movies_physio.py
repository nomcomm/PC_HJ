import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.stats import zscore
import subprocess
import argparse

# ==========================================
# CONFIGURATION
# ==========================================
GSR_DATA_DIR = 'data/01_processed_data/00_processed_GSR'
ECG_DATA_DIR = 'data/01_processed_data/01_processed_ECG'
CONDITIONS_FILE = 'data/sub_conditions.csv'
MOVIE_DIR = 'data/00_raw_data/99_movie_stimulus'
OUTPUT_FILE = 'data/visualizations/stimulus_comparison.mp4'
TEMP_DATA_VIDEO = 'data/visualizations/temp_data_animation.mp4'
OUTPUT_DIR = os.path.dirname(OUTPUT_FILE)

# Movie filenames
MOVIE_PC = os.path.join(MOVIE_DIR, 'Partly_Cloudy_Original.mov')
MOVIE_PCM = os.path.join(MOVIE_DIR, 'Partly_Cloudy_Modified.mov')

DURATION = 319  # seconds

# Visualization settings
# Target width: 1280 (matches 2x 640px movies)
# Target height: increased to accommodate 2x2 grid (e.g. 720px height) -> 7.2 inches
FIG_SIZE = (12.8, 7.2) 
DPI = 100
FPS = 24

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_conditions():
    print("Loading conditions...")
    try:
        conditions_df = pd.read_csv(CONDITIONS_FILE)
        conditions_df['SubjectID'] = conditions_df['RespondentName'].apply(lambda x: x.split('_')[0])
        return dict(zip(conditions_df['SubjectID'], conditions_df['SourceStimulus']))
    except Exception as e:
        print(f"Error loading conditions: {e}")
        return {}

def load_split_half_data(data_dir, condition_map, value_column, perform_zscore=False):
    print(f"Loading split-half data from {data_dir}...")
    files = glob.glob(os.path.join(data_dir, '*.csv'))
    # Sort files to ensure deterministic split
    files.sort()
    
    data = {'PC': [], 'PCM': []}
    
    for f in files:
        try:
            filename = os.path.basename(f)
            subject_id = filename.split('_')[0]
            condition = condition_map.get(subject_id)
            
            if not condition or condition not in data:
                continue
                
            df = pd.read_csv(f)
            
            if value_column in df.columns:
                signal = df[value_column].values
            else:
                possible_cols = [c for c in df.columns if 'Heart' in c or 'GSR' in c]
                if possible_cols:
                    signal = df[possible_cols[0]].values
                else:
                    continue

            # Ensure length is 638
            if len(signal) != 638:
                if len(signal) > 638:
                    signal = signal[:638]
                else:
                    pad_width = 638 - len(signal)
                    signal = np.pad(signal, (0, pad_width), constant_values=np.nan)

            if perform_zscore:
                if np.nanstd(signal) == 0:
                    continue
                signal = zscore(signal, nan_policy='omit')
                
            data[condition].append(signal)

        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    split_halves = {} # Structure: {'PC': (mean1, mean2), 'PCM': (mean1, mean2)}
    time_axis = np.linspace(0, DURATION, 638)
    
    for cond in ['PC', 'PCM']:
        if data[cond]:
            matrix = np.array(data[cond])
            n_subs = matrix.shape[0]
            if n_subs < 2:
                # Fallback if not enough subjects
                mean = np.nanmean(matrix, axis=0)
                split_halves[cond] = (mean, mean)
            else:
                mid = n_subs // 2
                half1 = matrix[:mid, :]
                half2 = matrix[mid:, :]
                mean1 = np.nanmean(half1, axis=0)
                mean2 = np.nanmean(half2, axis=0)
                split_halves[cond] = (mean1, mean2)
        else:
            zeros = np.zeros(638)
            split_halves[cond] = (zeros, zeros)
            
    return split_halves, time_axis

def generate_data_video(time_axis, gsr_splits, hr_splits, duration_limit=None):
    print("Generating data animation using Matplotlib (2x2 Grid)...")
    
    # Setup Figure: 2x2
    # Row 1: GSR (PC, PCM)
    # Row 2: hR (PC, PCM)
    # Share X axis
    # Share Y axis within rows? Yes, for comparison.
    
    # Order of subplots:
    # 1: Top Left (PC GSR)
    # 2: Top Right (PCM GSR)
    # 3: Bottom Left (PC HR)
    # 4: Bottom Right (PCM HR)
    
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE, sharex=True, dpi=DPI)
    plt.subplots_adjust(wspace=0.15, hspace=0.3, left=0.08, right=0.98, top=0.90, bottom=0.10)
    
    (ax_gsr_pc, ax_gsr_pcm), (ax_hr_pc, ax_hr_pcm) = axes
    
    # Calculate global limits per signal type for consistent Y-axis
    # GSR limits
    gsr_all = []
    for cond in ['PC', 'PCM']:
        gsr_all.extend([gsr_splits[cond][0], gsr_splits[cond][1]])
    gsr_all = np.concatenate(gsr_all)
    gsr_min, gsr_max = np.nanmin(gsr_all), np.nanmax(gsr_all)
    gsr_pad = 0.1 * (gsr_max - gsr_min)
    gsr_lims = (gsr_min - gsr_pad, gsr_max + gsr_pad)
    
    # HR limits
    hr_all = []
    for cond in ['PC', 'PCM']:
        hr_all.extend([hr_splits[cond][0], hr_splits[cond][1]])
    hr_all = np.concatenate(hr_all)
    hr_min, hr_max = np.nanmin(hr_all), np.nanmax(hr_all)
    hr_pad = 0.1 * (hr_max - hr_min)
    hr_lims = (hr_min - hr_pad, hr_max + hr_pad)
    
    color_1 = '#1f77b4' # Split 1
    color_2 = '#ff7f0e' # Split 2
    
    # --- PLOTTING ---
    
    # Helper to setup axis
    def setup_axis(ax, title, ylabel, ylims):
        ax.set_title(title, fontsize=10)
        ax.set_ylim(ylims)
        ax.set_xlim(0, DURATION)
        ax.grid(True, alpha=0.3)
        if ylabel:
            ax.set_ylabel(ylabel)
            
    # Top Left: PC GSR
    ax_gsr_pc.plot(time_axis, gsr_splits['PC'][0], label='Split 1', color=color_1, alpha=0.8)
    ax_gsr_pc.plot(time_axis, gsr_splits['PC'][1], label='Split 2', color=color_2, alpha=0.8)
    setup_axis(ax_gsr_pc, 'GSR - Original (PC)', 'GSR (Z)', gsr_lims)
    ax_gsr_pc.legend(loc='upper right', fontsize='x-small')
    
    # Top Right: PCM GSR
    ax_gsr_pcm.plot(time_axis, gsr_splits['PCM'][0], label='Split 1', color=color_1, alpha=0.8)
    ax_gsr_pcm.plot(time_axis, gsr_splits['PCM'][1], label='Split 2', color=color_2, alpha=0.8)
    ax_gsr_pcm.tick_params(labelleft=False) # Hide Y labels for right plot
    setup_axis(ax_gsr_pcm, 'GSR - Modified (PCM)', None, gsr_lims)
    
    # Bottom Left: PC HR
    ax_hr_pc.plot(time_axis, hr_splits['PC'][0], label='Split 1', color=color_1, alpha=0.8)
    ax_hr_pc.plot(time_axis, hr_splits['PC'][1], label='Split 2', color=color_2, alpha=0.8)
    setup_axis(ax_hr_pc, 'Heart Rate - Original (PC)', 'HR (BPM)', hr_lims)
    ax_hr_pc.set_xlabel('Time (s)')
    
    # Bottom Right: PCM HR
    ax_hr_pcm.plot(time_axis, hr_splits['PCM'][0], label='Split 1', color=color_1, alpha=0.8)
    ax_hr_pcm.plot(time_axis, hr_splits['PCM'][1], label='Split 2', color=color_2, alpha=0.8)
    ax_hr_pcm.tick_params(labelleft=False)
    setup_axis(ax_hr_pcm, 'Heart Rate - Modified (PCM)', None, hr_lims)
    ax_hr_pcm.set_xlabel('Time (s)')

    # --- ANIMATION ELEMENTS ---
    
    lines = []
    shades = []
    
    # Create line/shade for all 4 axes
    for ax in axes.flat:
        l = ax.axvline(x=0, color='red', linestyle='-', alpha=0.8)
        s = ax.axvspan(0, DURATION, color='gray', alpha=0.6, zorder=100)
        lines.append(l)
        shades.append(s)
    
    def init():
        return lines + shades

    def update(frame_num):
        t = frame_num / FPS
        if duration_limit and t > duration_limit:
            t = duration_limit
            
        # Update logic same as before, just loop
        for l in lines:
            l.set_xdata([t, t])
            
        for s in shades:
            xy = s.get_xy()
            xy[0, 0] = t # xmin
            xy[1, 0] = t # xmin
            xy[4, 0] = t # closing point
            s.set_xy(xy)
        
        return lines + shades

    limit = duration_limit if duration_limit else DURATION
    total_frames = int(limit * FPS)
    
    anim = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True)
    
    writer = FFMpegWriter(fps=FPS, metadata=dict(artist='Antigravity'), bitrate=3000) # Increased bitrate for larger video
    anim.save(TEMP_DATA_VIDEO, writer=writer)
    plt.close()
    print(f"Saved {TEMP_DATA_VIDEO}")

def composite_video(duration_limit=None):
    print("Compositing with FFMPEG...")
    
    # Need to verify if the 2x2 data grid (12.8" width = 1280px) still aligns with 
    # Side-by-side movies (640+640 = 1280px).
    # Yes, it should.
    
    cmd = [
        'ffmpeg', '-y',
        '-i', MOVIE_PC,
        '-i', MOVIE_PCM,
        '-i', TEMP_DATA_VIDEO,
        '-filter_complex',
        # Stack movies horizontally
        '[0:v][1:v]hstack=inputs=2[top];'
        # Stack Data video (which is already 2x2 grid) BELOW the movies
        '[top][2:v]vstack=inputs=2[outv];'
        # Mix audio
        '[0:a][1:a]amix=inputs=2:duration=shortest[outa]',
        
        '-map', '[outv]',
        '-map', '[outa]',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-shortest', 
        OUTPUT_FILE
    ]
    
    if duration_limit:
        pass # -shortest on inputs handles it usually if we trim inputs, for now rely on -shortest of inputs
    
    print("Running command:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"Saved final video to {OUTPUT_FILE}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run in test mode (10s)')
    args = parser.parse_args()

    ensure_dir(OUTPUT_DIR)
    
    # 1. Load Data
    cond_map = load_conditions()
    gsr_splits, time_axis = load_split_half_data(GSR_DATA_DIR, cond_map, 'GSR_Interpolated', True)
    hr_splits, _ = load_split_half_data(ECG_DATA_DIR, cond_map, 'Heart Rate PPG ALG', False)
    
    duration = 10 if args.test else DURATION
    
    # 2. Generate Data Video
    generate_data_video(time_axis, gsr_splits, hr_splits, duration_limit=duration)
    
    # 3. Composite
    composite_video(duration_limit=duration)
    
    print("Done!")

if __name__ == "__main__":
    main()
