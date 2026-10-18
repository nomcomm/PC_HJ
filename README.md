Analysis for Schmälzle, Cho & Turner (2026)
===========================================

Data and analysis code for *Schmälzle, R., Cho, H. J., & Turner, M. (2026). The Bio-Functional Audience: Shared Physiological Alignment to Narrative Media. Communication
Research.* [link](link_goes_here) · preprint: [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.02.05.703516v1)

***

<img align="right" width=400px src=scripts/explainer_fig.png>

Participants watched the Pixar short film *Partly Cloudy* (2009) in the lab while eye movements,
pupil size, heart rate and electrodermal activity were recorded. Half of the participants saw the
original film (`PC`), half a temporally scrambled version (`PCM`) whose physiological responses were
computationally unscrambled. We examine how strongly the audience aligns across these biological
channels (inter-subject correlation).

### Code

-   The notebooks to reproduce the analyses are in the scripts folder, numbered in the order in which they are run:
    -   `00_Preprocess_Channels_Integrate` – preprocessing of eye tracking, ECG and GSR and integration into one 20-Hz file per participant
    -   `01_Plot_Movie_Suppl` – supplementary movie figure
    -   `02_H1_Precompute`, `02a_H1_Precompute-Manipulated` – split-half alignment per channel (original / scrambled film)
    -   `03_H1_Plot_Figure2(_SPLITHALF)` – Figure 2
    -   `04_H2_H3_CommonDivergent_ISC_Precompute`, `05_RQ_Precompute`, `06_H2_H3_RQ_Plot_Figure3` – H2, H3, RQ and Figure 3
    -   `07_DynISC` – dynamic (sliding-window) ISC, Figure 4
    -   `08_ReverseCorr`, `08a_ReverseCorr-Manipulated` – reverse correlation
    -   `09_ISC_Channel_Relationships` – relationships between channels
    -   `10_PlotIndivdiual` – individual-level plots
-   Helper functions are in `scripts/utils/`.

### Data

-   The raw recordings (`data/00_raw_data`, `data/01_processed_data`) are too large to share.
-   `data/02_processed_combined_data/` – one file per participant at 20 Hz: `Time_sec`, `GSR`, `Heart_Rate`, `Pupil_Mean`, `Gaze_X`, `Gaze_Y` (screen pixels).
-   `data/sub_conditions.csv` – film version per participant (`PC` = original, `PCM` = scrambled) and prior exposure to the film (`Seen`).
-   `data/04_analysis/` – results of the notebooks (H1–H3, RQ, dynamic ISC, channel relationships).
-   The film itself is not included.

### Dependencies

-   Python 3.10 with numpy, pandas, scipy, matplotlib, seaborn, nltools, tqdm; opencv-python and moviepy for the movie visualizations; msuthemes for plot styling.


2026 | Schmälzle, Cho & Turner
