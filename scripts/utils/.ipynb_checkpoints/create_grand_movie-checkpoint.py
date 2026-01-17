
import cv2
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
from scipy.stats import zscore
import argparse
from pathlib import Path  # <--- Use Pathlib for robustness

# --- DYNAMIC PATH SETUP ---
try:
    # If running as a script file
    script_location = Path(__file__).resolve()
    # utils -> scripts -> root
    PROJECT_ROOT = script_location.parent.parent.parent
except NameError:
    # If running interactively (Jupyter) and __file__ is undefined
    # Assumes notebook is in 'scripts' folder
    PROJECT_ROOT = Path(os.getcwd()).parent

# Define Data Paths relative to Root
# Note: Using Path objects makes joining paths safer across OS
MOVIE_DIR = PROJECT_ROOT / "data/00_raw_data/99_movie_stimulus"
PC_MOVIE = MOVIE_DIR / "Partly_Cloudy_Original.mov"
PCM_MOVIE = MOVIE_DIR / "Partly_Cloudy_Modified.mov"

GSR_DIR = PROJECT_ROOT / "data/01_processed_data/00_processed_GSR"
ECG_DIR = PROJECT_ROOT / "data/01_processed_data/01_processed_ECG"
ET_DIR = PROJECT_ROOT / "data/01_processed_data/02_processed_ET"

CONDITIONS_FILE = PROJECT_ROOT / "data/sub_conditions.csv"
OUTPUT_VIDEO_DIR = PROJECT_ROOT / "data/03_visualizations"
OUTPUT_VIDEO = OUTPUT_VIDEO_DIR / "grand_integrated_movie.mp4"

# Ensure output directory exists
os.makedirs(str(OUTPUT_VIDEO_DIR), exist_ok=True)

# Movie & Layout Config
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
QUADRANT_W = 960  # 1920 / 2
QUADRANT_H = 540  # 1080 / 2

# Plot Visuals
PLOT_BG_COLOR = (255, 255, 255)
CURSOR_COLOR = (0, 0, 0)
PLOT_MARGIN_LEFT = 0.10
PLOT_MARGIN_RIGHT = 0.98
PLOT_MARGIN_TOP = 0.95
PLOT_MARGIN_BOTTOM = 0.05

# Gaze Overlay Config
GAZE_DOT_COLOR_MEAN = (0, 0, 255)       # Red (BGR)
GAZE_DOT_RADIUS_MEAN = 15
GAZE_DOT_COLOR_S1 = (255, 0, 0)         # Blue (BGR)
GAZE_DOT_COLOR_S2 = (0, 165, 255)       # Orange (BGR)
GAZE_DOT_RADIUS_SUBMEAN = 12
GAZE_DOT_RADIUS_INDIV = 4
GAZE_DOT_ALPHA_INDIV = 0.4

# Data Processing Constants
DURATION_SEC = 319
SAMPLES_PHYSIO = 638  

def load_conditions():
    cond_map = {}
    if os.path.exists(CONDITIONS_FILE):
        try:
            cond_df = pd.read_csv(CONDITIONS_FILE)
            if 'SubjectID' not in cond_df.columns:
                cond_df['SubjectID'] = cond_df['RespondentName'].apply(lambda x: str(x).split('_')[0])
            
            for _, row in cond_df.iterrows():
                sid = str(row['SubjectID'])
                if 'SourceStimulus' in row:
                    cond_map[sid] = row['SourceStimulus']
                elif 'condition' in row:
                    cond_map[sid] = row['condition']
        except Exception as e:
            print(f"Error loading conditions: {e}")
    else:
        print(f"WARNING: Conditions file not found at {CONDITIONS_FILE}")
    return cond_map

def load_gsr(cond_map):
    print(f"Loading GSR data from {GSR_DIR}...")
    files = glob.glob(os.path.join(GSR_DIR, '*.csv'))
    files.sort()
    
    data = {'PC': [], 'PCM': []}
    
    for f in files:
        sub_id = os.path.basename(f).split('_')[0]
        cond = cond_map.get(sub_id)
        if cond not in ['PC', 'PCM']: continue
        
        try:
            df = pd.read_csv(f)
            if 'GSR_Interpolated' not in df.columns: continue
            
            sig = df['GSR_Interpolated'].values
            
            if len(sig) > 638: sig = sig[:638]
            elif len(sig) < 638: sig = np.pad(sig, (0, 638 - len(sig)), constant_values=np.nan)
            
            if np.nanstd(sig) > 0:
                sig = zscore(sig, nan_policy='omit')
                data[cond].append(sig)
        except: pass
        
    return data

def load_hr(cond_map):
    print(f"Loading HR data from {ECG_DIR}...")
    files = glob.glob(os.path.join(ECG_DIR, '*.csv'))
    files.sort()
    
    data = {'PC': [], 'PCM': []}
    
    for f in files:
        sub_id = os.path.basename(f).split('_')[0]
        cond = cond_map.get(sub_id)
        if cond not in ['PC', 'PCM']: continue
        
        try:
            df = pd.read_csv(f)
            col = 'Heart_Rate'
            if col not in df.columns:
                 poss = [c for c in df.columns if 'Heart' in c]
                 if poss: col = poss[0]
                 else: continue
            
            sig = df[col].values
            
            if len(sig) > 638: sig = sig[:638]
            elif len(sig) < 638: sig = np.pad(sig, (0, 638 - len(sig)), constant_values=np.nan)
            
            mask = (sig >= 40) & (sig <= 130)
            sig = np.where(mask, sig, np.nan)
            
            if np.isnan(sig).sum() <= (len(sig) * 0.5): 
                data[cond].append(sig)
        except: pass
        
    return data

def load_et(cond_map):
    print(f"Loading ET data from {ET_DIR}...")
    files = glob.glob(os.path.join(ET_DIR, '*_processed_et.csv'))
    files.sort()
    
    data = {'PC': [], 'PCM': []}
    
    for f in files:
        sub_id = os.path.basename(f).split('_')[0]
        cond = cond_map.get(sub_id)
        if cond not in ['PC', 'PCM']: continue
        
        try:
            df = pd.read_csv(f)
            if len(df) < 100: continue
            
            cols = ['Mean_Pupil', 'Interpolated Gaze X', 'Interpolated Gaze Y']
            if not all(c in df.columns for c in cols): continue
            
            df['idx'] = np.linspace(0, 1, len(df))
            new_idx = np.linspace(0, 1, 638)
            
            resampled = pd.DataFrame(index=new_idx)
            for c in cols:
                resampled[c] = np.interp(new_idx, df['idx'], df[c])
                
            data[cond].append(resampled)
        except: pass

    return data

def get_split_halves(data_list):
    if not data_list: return None, None
    n = len(data_list)
    if n < 2: return data_list, data_list 
    mid = n // 2
    return data_list[:mid], data_list[mid:]

def compute_mean_traces(data_list, et_mode=False):
    if not data_list: return np.full((638, 3) if et_mode else (638,), np.nan)
    
    if et_mode:
        stack = np.stack([df.values for df in data_list]) # (N, 638, 3)
        return np.nanmean(stack, axis=0)
    else:
        stack = np.array(data_list)
        return np.nanmean(stack, axis=0)

def generate_static_plot_panel(gsr_s1, gsr_s2, hr_s1, hr_s2, et_s1, et_s2, title):
    """
    Generates the tall 5-row plot panel.
    """
    # Fix scope issues: Pass target width/height explicitly to resize logic
    target_w = int(QUADRANT_W)
    target_h = int(QUADRANT_H)
    
    fig, axes = plt.subplots(5, 1, figsize=(target_w/100, target_h/100), dpi=100, sharex=True)
    fig.subplots_adjust(left=PLOT_MARGIN_LEFT, right=PLOT_MARGIN_RIGHT, top=PLOT_MARGIN_TOP, bottom=PLOT_MARGIN_BOTTOM)
    
    x = np.linspace(0, DURATION_SEC, 638)
    
    # Handle NaN arrays to prevent plot crashes
    def clean(arr): return arr if arr is not None else np.full_like(x, np.nan)
    
    plots = [
        (clean(gsr_s1), clean(gsr_s2), 'GSR (Z)', 'C0', 'C1', False),
        (clean(hr_s1), clean(hr_s2), 'Heart Rate', 'C0', 'C1', False),
        (clean(et_s1)[:, 0] if et_s1 is not None else np.nan, clean(et_s2)[:, 0] if et_s2 is not None else np.nan, 'Pupil Size', 'C0', 'C1', False),
        (clean(et_s1)[:, 1] if et_s1 is not None else np.nan, clean(et_s2)[:, 1] if et_s2 is not None else np.nan, 'Gaze X', 'C0', 'C1', False),
        (clean(et_s1)[:, 2] if et_s1 is not None else np.nan, clean(et_s2)[:, 2] if et_s2 is not None else np.nan, 'Gaze Y', 'C0', 'C1', True)
    ]
    
    def render(alpha_factor=1.0):
        for i, (d1, d2, label, c1, c2, invert) in enumerate(plots):
            ax = axes[i]
            ax.clear()
            ax.set_ylabel(label, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, DURATION_SEC)
            ax.set_yticks([]) # Hide Y-ticks to save space
            
            if invert: ax.invert_yaxis()
            
            base_alpha = 0.8
            ax.plot(x, d1, color=c1, alpha=base_alpha * alpha_factor, label='Split 1', linewidth=1.5)
            ax.plot(x, d2, color=c2, alpha=base_alpha * alpha_factor, label='Split 2', linewidth=1.5)
            
            if i == 0:
                ax.set_title(title, fontsize=12, fontweight='bold')
                
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        image = np.frombuffer(raw_data, dtype=np.uint8).reshape(size[1], size[0], 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return cv2.resize(image, (target_w, target_h))

    opaque = render(1.0)
    dimmed = render(0.3)
    
    plt.close(fig)
    return dimmed, opaque

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help="Run only 10 seconds")
    # Fake args if running in Jupyter
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    
    print(f"--- Starting Integrated Movie Generation ---")
    print(f"Project Root: {PROJECT_ROOT}")
    
    if not os.path.exists(PC_MOVIE):
        print(f"ERROR: Movie file not found: {PC_MOVIE}")
        return

    # 1. Load Data
    cond_map = load_conditions()
    gsr_data = load_gsr(cond_map)
    hr_data = load_hr(cond_map)
    et_data = load_et(cond_map)
    
    # 2. Prepare Split Averages
    proc = {'PC': {}, 'PCM': {}}
    
    for cond in ['PC', 'PCM']:
        g1, g2 = get_split_halves(gsr_data[cond])
        proc[cond]['gsr_s1'] = compute_mean_traces(g1)
        proc[cond]['gsr_s2'] = compute_mean_traces(g2)
        
        h1, h2 = get_split_halves(hr_data[cond])
        proc[cond]['hr_s1'] = compute_mean_traces(h1)
        proc[cond]['hr_s2'] = compute_mean_traces(h2)
        
        e1, e2 = get_split_halves(et_data[cond])
        proc[cond]['et_s1'] = compute_mean_traces(e1, et_mode=True)
        proc[cond]['et_s2'] = compute_mean_traces(e2, et_mode=True)
        
        # Individual clouds (handle empty lists safely)
        proc[cond]['cloud_s1'] = [df.values[:, 1:3] for df in e1] if e1 else []
        proc[cond]['cloud_s2'] = [df.values[:, 1:3] for df in e2] if e2 else []

    # 3. Generate Static Plot Panels
    print("Generating static plot panels...")
    panels = {}
    for cond in ['PC', 'PCM']:
        dim, opq = generate_static_plot_panel(
            proc[cond]['gsr_s1'], proc[cond]['gsr_s2'],
            proc[cond]['hr_s1'], proc[cond]['hr_s2'],
            proc[cond]['et_s1'], proc[cond]['et_s2'],
            f"Psychophysiology - {cond}"
        )
        panels[cond] = (dim, opq)

    # 4. Video Composition Loop
    cap_pc = cv2.VideoCapture(str(PC_MOVIE))
    cap_pcm = cv2.VideoCapture(str(PCM_MOVIE))
    
    fps = cap_pc.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_pc.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if args.test:
        total_frames = int(fps * 10) 
        
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*'mp4v'), fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    
    x_start_px = int(QUADRANT_W * PLOT_MARGIN_LEFT)
    x_end_px = int(QUADRANT_W * PLOT_MARGIN_RIGHT)
    axis_width = x_end_px - x_start_px
    
    print(f"Processing {total_frames} frames...")
    
    for i in tqdm(range(total_frames)):
        ret1, frame_pc = cap_pc.read()
        ret2, frame_pcm = cap_pcm.read()
        
        if not ret1 or not ret2: break
        
        frame_pc = cv2.resize(frame_pc, (QUADRANT_W, QUADRANT_H))
        frame_pcm = cv2.resize(frame_pcm, (QUADRANT_W, QUADRANT_H))
        
        time_sec = i / fps
        sample_idx = int((time_sec / DURATION_SEC) * SAMPLES_PHYSIO)
        if sample_idx >= SAMPLES_PHYSIO: sample_idx = SAMPLES_PHYSIO - 1
        
        # --- DRAW GAZE OVERLAYS ---
        for cond, frame in [('PC', frame_pc), ('PCM', frame_pcm)]:
            # Cloud
            for key, color in [('cloud_s1', GAZE_DOT_COLOR_S1), ('cloud_s2', GAZE_DOT_COLOR_S2)]:
                for trace in proc[cond][key]:
                    if sample_idx < len(trace):
                        pt = trace[sample_idx]
                        if not np.isnan(pt).any():
                            # Assume data is normalized 1920x1080, scale to 960x540
                            sx, sy = int(pt[0]*0.5), int(pt[1]*0.5)
                            cv2.circle(frame, (sx, sy), GAZE_DOT_RADIUS_INDIV, color, -1)
            
            # Sub-Means
            for key, color in [('et_s1', GAZE_DOT_COLOR_S1), ('et_s2', GAZE_DOT_COLOR_S2)]:
                trace = proc[cond][key] # (638, 3)
                if trace is not None and not np.isnan(trace).all():
                    pt = trace[sample_idx, 1:3]
                    if not np.isnan(pt).any():
                        sx, sy = int(pt[0]*0.5), int(pt[1]*0.5)
                        cv2.circle(frame, (sx, sy), GAZE_DOT_RADIUS_SUBMEAN, color, 2)
            
            # Grand Mean
            p1 = proc[cond]['et_s1']
            p2 = proc[cond]['et_s2']
            if p1 is not None and p2 is not None:
                pt1 = p1[sample_idx, 1:3]
                pt2 = p2[sample_idx, 1:3]
                gm = np.nanmean([pt1, pt2], axis=0)
                if not np.isnan(gm).any():
                    sx, sy = int(gm[0]*0.5), int(gm[1]*0.5)
                    cv2.circle(frame, (sx, sy), GAZE_DOT_RADIUS_MEAN, GAZE_DOT_COLOR_MEAN, -1)

        # --- PREPARE PLOT PANELS ---
        progress = time_sec / DURATION_SEC
        cursor_x = int(x_start_px + progress * axis_width)
        cursor_x = max(x_start_px, min(cursor_x, x_end_px))
        
        plot_frames = []
        for cond in ['PC', 'PCM']:
            dim, opq = panels[cond]
            p_img = dim.copy()
            if cursor_x > 0:
                # Reveal effect
                p_img[:, :cursor_x] = opq[:, :cursor_x]
            
            cv2.line(p_img, (cursor_x, 0), (cursor_x, QUADRANT_H), CURSOR_COLOR, 2)
            plot_frames.append(p_img)
            
        # --- STACK FINAL FRAME ---
        top_row = np.hstack((frame_pc, frame_pcm))
        bottom_row = np.hstack((plot_frames[0], plot_frames[1]))
        full_frame = np.vstack((top_row, bottom_row))
        
        writer.write(full_frame)

    cap_pc.release()
    cap_pcm.release()
    writer.release()
    print(f"Saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()