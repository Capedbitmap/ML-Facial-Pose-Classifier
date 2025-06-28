import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import json

# --- Configuration ---
DATA_DIR = "enriched_processed_data"
RESULTS_DIR = "Results"

# Attempt to load optimized thresholds dynamically; otherwise fall back to
# previously tuned constants.
THRESHOLD_FILE = "optimized_thresholds.json"

if os.path.exists(THRESHOLD_FILE):
    with open(THRESHOLD_FILE, "r") as fp:
        BEST_THRESHOLDS = json.load(fp)
    print(f"Loaded optimized thresholds from '{THRESHOLD_FILE}'.")
else:
    BEST_THRESHOLDS = {
        'hand_raise_thresh': 0.2181,
        'ear_thresh': 0.3654,
        'drowsy_frames': 51,
        'mar_thresh': 0.4001,
        'yawn_frames': 5,
        'gaze_left_thresh': 0.4483,
        'gaze_right_thresh': 0.5746,
        'pose_h_left_thresh': -0.5296,
        'pose_h_right_thresh': 0.1512,
        'pose_forward_thresh': 0.2358,
        'pose_v_up_thresh': 0.35,
        'pose_v_down_thresh': 0.65,
        'smile_thresh': -0.0010,
    }
    print("Using fallback threshold constants (optimized file not found).")

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Results will be saved in the '{RESULTS_DIR}/' directory.")

# --- 1. Load and Prepare All Data ---
all_csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
df_list = []
for f in all_csv_files:
    temp_df = pd.read_csv(f)
    subject_id = os.path.basename(f).split('_')[0]
    temp_df['subject_id'] = f"Subject_{subject_id}"
    df_list.append(temp_df)
df = pd.concat(df_list, ignore_index=True)
print(f"Loaded and combined data for {df['subject_id'].nunique()} subjects.")

# --- Optional: replicate optimiser's cleaning so global metrics match its accuracy numbers ---
TRIM_PERCENTAGE = 0.1  # match 2_optimize_thresholds.py
IQR_MULTIPLIER = 3

def clean_data_for_metrics(df_all: pd.DataFrame, csv_files: list[str]):
    """Replicate the trimming/outlier-removal logic from 2_optimize_thresholds.py so
    that performance metrics align with the optimiser's printed accuracies."""

    df = df_all.copy()

    # Map each source CSV file to rows so we know where scenario segments live
    df['video_file'] = ''
    start_idx = 0
    for f in csv_files:
        try:
            n_rows = len(pd.read_csv(f))
            end_idx = start_idx + n_rows
            df.loc[start_idx:end_idx - 1, 'video_file'] = os.path.basename(f)
            start_idx = end_idx
        except Exception as e:
            print(f"Warning: could not load {f} during cleaning: {e}")

    state_to_metric = {
        "Drowsy": "ear", "Yawning": "mar", "Gaze Left": "gaze_avg_ratio",
        "Gaze Right": "gaze_avg_ratio", "Gaze Center": "gaze_avg_ratio",
        "Looking Left": "pose_horiz_centered", "Looking Right": "pose_horiz_centered",
        "Hand Raise": "hand_raise_metric", "Forward Pose": "pose_forward_metric",
        "Emotion Happy": "mouth_curl_metric", "Emotion Sad": "mouth_curl_metric",
        "Emotion Surprise": "emotion_surprise_metric",
    }

    rows_before = len(df)
    idx_to_drop = set()

    for (video, scenario), grp in df.groupby(['video_file', 'scenario_ground_truth']):
        if scenario not in state_to_metric:
            continue
        metric_col = state_to_metric[scenario]
        n_frames = len(grp)
        trim_cnt = int(n_frames * TRIM_PERCENTAGE)

        core_grp = grp.iloc[trim_cnt: n_frames - trim_cnt]
        # mark first/last for drop
        idx_to_drop.update(grp.head(trim_cnt).index)
        idx_to_drop.update(grp.tail(trim_cnt).index)

        valid_metric = core_grp[metric_col].dropna()
        if len(valid_metric) > 4:
            Q1, Q3 = valid_metric.quantile(0.25), valid_metric.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                low_bnd, up_bnd = Q1 - IQR_MULTIPLIER * IQR, Q3 + IQR_MULTIPLIER * IQR
                outlier_mask = (core_grp[metric_col] < low_bnd) | (core_grp[metric_col] > up_bnd)
                idx_to_drop.update(core_grp[outlier_mask].index)

    cleaned_df = df.drop(index=list(idx_to_drop)).copy()
    print(f"Data cleaning for metrics: removed {rows_before - len(cleaned_df)} frames; {len(cleaned_df)} remain.")
    return cleaned_df

# --- 2. The Main Reporting Function ---
def generate_and_save_results(data, output_dir, title_prefix, states_subset=None, produce_cm=True, produce_roc=True):
    os.makedirs(output_dir, exist_ok=True)
    
    # --- A. Generate Final Binary Predictions ---
    predictions = {}
    drowsy_signal = data['ear'] < BEST_THRESHOLDS['ear_thresh']
    drowsy_streak = drowsy_signal.rolling(window=BEST_THRESHOLDS['drowsy_frames'], min_periods=1).sum()
    predictions['Drowsy'] = (drowsy_streak >= BEST_THRESHOLDS['drowsy_frames'])
    
    yawn_signal = data['mar'] > BEST_THRESHOLDS['mar_thresh']
    yawn_streak = yawn_signal.rolling(window=BEST_THRESHOLDS['yawn_frames'], min_periods=1).sum()
    predictions['Yawning'] = (yawn_streak >= BEST_THRESHOLDS['yawn_frames'])

    predictions['Hand Raise'] = data['hand_raise_metric'] > BEST_THRESHOLDS['hand_raise_thresh']

    # --- Gaze related ---
    predictions['Gaze Left'] = data['gaze_avg_ratio'] < BEST_THRESHOLDS['gaze_left_thresh']
    predictions['Gaze Right'] = data['gaze_avg_ratio'] > BEST_THRESHOLDS['gaze_right_thresh']
    predictions['Gaze Center'] = (
        (data['gaze_avg_ratio'] >= BEST_THRESHOLDS['gaze_left_thresh']) &
        (data['gaze_avg_ratio'] <= BEST_THRESHOLDS['gaze_right_thresh'])
    )

    # --- Head-pose related ---
    predictions['Looking Left'] = data['pose_horiz_centered'] < BEST_THRESHOLDS['pose_h_left_thresh']
    predictions['Looking Right'] = data['pose_horiz_centered'] > BEST_THRESHOLDS['pose_h_right_thresh']
    predictions['Forward Pose'] = data['pose_forward_metric'] < BEST_THRESHOLDS['pose_forward_thresh']
    # Vertical head pose
    if 'pose_vert_ratio' in data.columns:
        predictions['Looking Up'] = data['pose_vert_ratio'] < BEST_THRESHOLDS.get('pose_v_up_thresh', 0.35)
        predictions['Looking Down'] = data['pose_vert_ratio'] > BEST_THRESHOLDS.get('pose_v_down_thresh', 0.65)

    # --- Smile detection ---
    if 'mouth_curl_metric' in data.columns:
        predictions['Smile'] = data['mouth_curl_metric'] < BEST_THRESHOLDS.get('smile_thresh', -0.0010)

    predictions_df = pd.DataFrame(predictions)
    
    # Prioritise mutually exclusive or more reliable states first so they aren't masked by broader detections
    STATE_PRIORITY = [
        'Yawning',
        'Hand Raise',
        'Smile',
        'Drowsy',
        'Forward Pose',
        'Looking Left',
        'Looking Right',
        'Looking Up',
        'Looking Down',
        'Gaze Left',
        'Gaze Right',
        'Gaze Center',
    ]

    # Ensure predictions contain these keys in that order & keep only recognised states
    if states_subset is None:
        all_possible_states = [s for s in STATE_PRIORITY if s in predictions]
    else:
        all_possible_states = [s for s in STATE_PRIORITY if s in predictions and s in states_subset]

    # --- B. Derive single predicted label per frame ---
    def _select_predicted_state(row):
        for st in all_possible_states:  # iterate in priority order
            if row.get(st, False):
                return st
        return 'None'

    predicted_labels = predictions_df.apply(_select_predicted_state, axis=1)
    # Map underlying emotion label to our Smile state
    true_raw = data['scenario_ground_truth'].replace({'Emotion Happy': 'Smile'})
    true_labels = true_raw.where(true_raw.isin(all_possible_states), 'None')

    # Filter rows where BOTH the ground-truth **and** the prediction are within the
    # recognised state list so that frames with a "None" prediction are ignored
    # in *both* the performance table and the confusion matrix. This keeps the
    # two artefacts consistent and avoids counting missed detections.
    valid_mask = (
        true_labels.isin(all_possible_states) &
        predicted_labels.isin(all_possible_states)
    )
    y_true_full = true_labels[valid_mask]
    y_pred_full = predicted_labels[valid_mask]

    if y_true_full.empty:
        print(f"  -> No valid data to generate a report for {title_prefix}. Skipping.")
        return

    # --- C. Unified Performance Metrics Table (precision/recall/F1, + overall) ---
    report_dict = classification_report(
        y_true_full,
        y_pred_full,
        labels=all_possible_states,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    perf_path = os.path.join(output_dir, f"{title_prefix}_performance_metrics.csv")
    report_df.to_csv(perf_path, float_format='%.3f')
    print(f"  -> Saved performance metrics table to {perf_path}")

    # --- D. High-quality Multi-class Confusion Matrix ---
    if produce_cm and len(all_possible_states) > 1:
        cm_mask = y_pred_full.isin(all_possible_states)
        y_true_cm = y_true_full[cm_mask]
        y_pred_cm = y_pred_full[cm_mask]

        combined_cm = confusion_matrix(y_true_cm, y_pred_cm, labels=all_possible_states)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            combined_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=all_possible_states,
            yticklabels=all_possible_states,
        )
        plt.title(f'{title_prefix} – Confusion Matrix')
        plt.ylabel('Actual State')
        plt.xlabel('Predicted State')
        cm_path = os.path.join(output_dir, f"{title_prefix}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved confusion matrix to {cm_path}")

    # --- E. Single Micro-average ROC Curve & AUC ---
    if produce_roc and len(all_possible_states) > 1:
        score_map = {
            'Drowsy': (1 - data['ear']),
            'Yawning': data['mar'],
            'Hand Raise': data['hand_raise_metric'],
            'Gaze Left': (1 - data['gaze_avg_ratio']),
            'Gaze Right': data['gaze_avg_ratio'],
            'Gaze Center': (-abs(data['gaze_avg_ratio'] - ((BEST_THRESHOLDS['gaze_left_thresh'] + BEST_THRESHOLDS['gaze_right_thresh']) / 2))),
            'Looking Left': (-data['pose_horiz_centered']),  # more negative => higher score
            'Looking Right': data['pose_horiz_centered'],
            'Forward Pose': (-data['pose_forward_metric']),
            'Looking Up': (-data['pose_vert_ratio']) if 'pose_vert_ratio' in data.columns else None,
            'Looking Down': data['pose_vert_ratio'] if 'pose_vert_ratio' in data.columns else None,
            'Smile': (-data['mouth_curl_metric']) if 'mouth_curl_metric' in data.columns else None,
        }

        from sklearn.preprocessing import label_binarize

        scores_df = pd.DataFrame(score_map).fillna(0)
        # Align score rows with the subset used for y_true_full
        scores_df = scores_df.loc[y_true_full.index]
        # Ensure columns align with states list
        scores_matrix = scores_df[all_possible_states].to_numpy()

        # Binarize true labels for ROC-AUC computation
        y_true_bin = label_binarize(y_true_full, classes=all_possible_states)

        if y_true_bin.shape[1] != scores_matrix.shape[1]:
            print("  -> Warning: mismatch between binarized labels and scores; skipping ROC generation.")
        else:
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), scores_matrix.ravel())
            roc_auc_val = auc(fpr, tpr)

            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Micro-average ROC (AUC = {roc_auc_val:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{title_prefix} – Micro-average ROC Curve')
            plt.legend(loc='lower right')
            roc_path = os.path.join(output_dir, f"{title_prefix}_roc_curve_micro.png")
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  -> Saved ROC curve to {roc_path} (AUC = {roc_auc_val:.3f})")

# --- 4. Generate Global and Per-Subject Reports (silo-wise) ---
print("\n--- Generating Global Results (per silo) ---")

# --- Generate global report for each silo ---
SILO_DEFINITIONS = {
    "HeadPoseHoriz": ['Forward Pose', 'Looking Left', 'Looking Right'],
    "HeadPoseVert": ['Looking Up', 'Looking Down'],
    "GazeDirection": ['Gaze Left', 'Gaze Right', 'Gaze Center'],
    "Smile": ['Smile'],
    "Yawning": ['Yawning'],
    "HandRaise": ['Hand Raise'],
    "Drowsy": ['Drowsy'],
}

cleaned_df = clean_data_for_metrics(df, all_csv_files)

for silo_name, states_list in SILO_DEFINITIONS.items():
    silo_title = f"Global_{silo_name}"
    generate_and_save_results(cleaned_df, os.path.join(RESULTS_DIR, "Global"), silo_title, states_subset=states_list)

# Generate combined global performance metrics (no CM/ROC)
print("\n--- Generating Combined Global Performance Metrics ---")
generate_and_save_results(cleaned_df, os.path.join(RESULTS_DIR, "Global"), "Global_AllStates", produce_cm=False, produce_roc=False)

print("\n--- Generating Per-Subject Results (per silo) ---")
for subject_id, subject_df in cleaned_df.groupby('subject_id'):
    print(f"- Processing {subject_id}...")

    for silo_name, states_list in SILO_DEFINITIONS.items():
        silo_title = f"{subject_id}_{silo_name}"
        generate_and_save_results(subject_df, os.path.join(RESULTS_DIR, subject_id), silo_title, states_subset=states_list)

print("\nReport generation complete!")

# =============================================================
#   Multi-label global evaluation across ALL states
# =============================================================

def build_prediction_flags(data):
    """Return a dict of boolean Series for every state using global thresholds."""
    preds = {}
    # Drowsy
    drowsy_signal = data['ear'] < BEST_THRESHOLDS['ear_thresh']
    drowsy_streak = drowsy_signal.rolling(window=BEST_THRESHOLDS['drowsy_frames'], min_periods=1).sum()
    preds['Drowsy'] = (drowsy_streak >= BEST_THRESHOLDS['drowsy_frames'])
    # Yawn
    yawn_signal = data['mar'] > BEST_THRESHOLDS['mar_thresh']
    yawn_streak = yawn_signal.rolling(window=BEST_THRESHOLDS['yawn_frames'], min_periods=1).sum()
    preds['Yawning'] = (yawn_streak >= BEST_THRESHOLDS['yawn_frames'])
    # Hand raise
    preds['Hand Raise'] = data['hand_raise_metric'] > BEST_THRESHOLDS['hand_raise_thresh']
    # Smile
    if 'mouth_curl_metric' in data.columns:
        preds['Smile'] = data['mouth_curl_metric'] < BEST_THRESHOLDS.get('smile_thresh', -0.0010)
    # Head pose horiz/vert & forward
    preds['Looking Left'] = data['pose_horiz_centered'] < BEST_THRESHOLDS['pose_h_left_thresh']
    preds['Looking Right'] = data['pose_horiz_centered'] > BEST_THRESHOLDS['pose_h_right_thresh']
    preds['Forward Pose'] = data['pose_forward_metric'] < BEST_THRESHOLDS['pose_forward_thresh']
    if 'pose_vert_ratio' in data.columns:
        preds['Looking Up'] = data['pose_vert_ratio'] < BEST_THRESHOLDS.get('pose_v_up_thresh', 0.35)
        preds['Looking Down'] = data['pose_vert_ratio'] > BEST_THRESHOLDS.get('pose_v_down_thresh', 0.65)
    # Gaze
    preds['Gaze Left'] = data['gaze_avg_ratio'] < BEST_THRESHOLDS['gaze_left_thresh']
    preds['Gaze Right'] = data['gaze_avg_ratio'] > BEST_THRESHOLDS['gaze_right_thresh']
    preds['Gaze Center'] = (
        (data['gaze_avg_ratio'] >= BEST_THRESHOLDS['gaze_left_thresh']) &
        (data['gaze_avg_ratio'] <= BEST_THRESHOLDS['gaze_right_thresh'])
    )
    return preds


def compute_global_multilabel_metrics(df_all: pd.DataFrame):
    """Compute per-state detection accuracies (recall on that class) plus an overall
    micro-average accuracy, and save to CSV so it mirrors the textual table that
    2_optimize_thresholds.py prints at the end of optimisation."""

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    preds = build_prediction_flags(df_all)

    # Optional mapping for ground-truth labels that differ from the prediction key
    alias = {"Smile": "Emotion Happy"}

    report_rows = {}

    # 1️⃣ Per-state metrics
    for state, pred_series in preds.items():
        gt_label = alias.get(state, state)
        mask_pos = (df_all['scenario_ground_truth'] == gt_label)
        if mask_pos.sum() == 0:
            continue

        acc_state = accuracy_score(np.ones(mask_pos.sum(), dtype=int), pred_series[mask_pos].astype(int))

        p_s, r_s, f_s, sup_s = precision_recall_fscore_support(
            np.ones(mask_pos.sum(), dtype=int), pred_series[mask_pos].astype(int), zero_division=0, average='binary')

        report_rows[state] = {
            'accuracy': acc_state,
            'precision': p_s,
            'recall': r_s,
            'f1-score': f_s,
            'support': sup_s,
        }

    # 2️⃣ Global (micro-average) metrics
    states = list(preds.keys())
    y_true_bin = np.zeros((len(df_all), len(states)), dtype=int)
    y_pred_bin = np.zeros_like(y_true_bin)

    for i, s in enumerate(states):
        true_label = alias.get(s, s)
        y_true_bin[:, i] = (df_all['scenario_ground_truth'] == true_label).astype(int)
        y_pred_bin[:, i] = preds[s].astype(int)

    # Compute precision/recall/F1 for each column (state)
    for i, state in enumerate(states):
        gt_label = alias.get(state, state)

        # Per-state accuracy evaluated only on frames where that state is the GT (matches optimiser logic)
        mask_pos = (df_all['scenario_ground_truth'] == gt_label)
        if mask_pos.sum() == 0:
            continue

        acc_state = accuracy_score(np.ones(mask_pos.sum(), dtype=int), preds[state][mask_pos].astype(int))

        p_s, r_s, f_s, sup_s = precision_recall_fscore_support(
            y_true_bin[:, i], y_pred_bin[:, i], zero_division=0, average='binary')

        report_rows[state] = {
            'accuracy': acc_state,
            'precision': p_s,
            'recall': r_s,
            'f1-score': f_s,
            'support': sup_s,
        }

    # 2️⃣ Global (micro-average) metrics
    p_g, r_g, f_g, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average='micro', zero_division=0)
    global_acc = accuracy_score(y_true_bin.reshape(-1), y_pred_bin.reshape(-1))

    report_rows['GLOBAL'] = {
        'accuracy': global_acc,
        'precision': p_g,
        'recall': r_g,
        'f1-score': f_g,
        'support': y_true_bin.shape[0],
    }

    # --- Save ---
    df_report = pd.DataFrame(report_rows).T.sort_index()
    out_path = os.path.join(RESULTS_DIR, 'Global', 'Global_Multilabel_Metrics.csv')
    df_report.to_csv(out_path, float_format='%.3f')
    print(f"\nGlobal multi-label accuracies saved to {out_path}")

# Already cleaned earlier to create 'cleaned_df'; reuse it
compute_global_multilabel_metrics(cleaned_df)

# =============================================================
#   Per-video accuracies (one row per video, accuracy per state)
# =============================================================

print("\n--- Computing per-video accuracy table ---")

def compute_accuracy_table_per_video(df_all: pd.DataFrame, output_path: str):
    """Create a CSV with one row per video and accuracy for every state column."""

    preds_bool_all = build_prediction_flags(df_all)
    states = list(preds_bool_all.keys())

    alias = {"Smile": "Emotion Happy"}

    per_video_records = []
    for video_name, vid_rows in df_all.groupby('video_file'):
        row_res = {"video": video_name}
        preds_bool = {s: series[vid_rows.index] for s, series in preds_bool_all.items()}

        for state in states:
            gt_label = alias.get(state, state)
            mask = vid_rows['scenario_ground_truth'] == gt_label
            if mask.sum() == 0:
                acc = np.nan  # or 0.0 if numbers strictly required
            else:
                acc = (preds_bool[state][mask]).mean()  # proportion of correct positives
            row_res[state] = acc

        per_video_records.append(row_res)

    df_out = pd.DataFrame(per_video_records).set_index('video')
    df_out.to_csv(output_path, float_format='%.3f')
    print(f"Per-video accuracy table saved to {output_path}")


per_video_output = os.path.join(RESULTS_DIR, 'Global', 'Global_PerVideo_Accuracy.csv')
compute_accuracy_table_per_video(cleaned_df, per_video_output)

print("\nAll reporting complete!")
