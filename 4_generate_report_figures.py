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

# --- 2. The Main Reporting Function ---
def generate_and_save_results(data, output_dir, title_prefix, states_subset=None):
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

    # Filter rows belonging to known states so metrics are meaningful
    valid_mask = true_labels.isin(all_possible_states)
    y_true_full = true_labels[valid_mask]
    y_pred_full = predicted_labels[valid_mask].replace({'None': 'No Prediction'})

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
    extended_labels = all_possible_states + ['No Prediction']
    combined_cm = confusion_matrix(y_true_full, y_pred_full, labels=extended_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        combined_cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=extended_labels,
        yticklabels=extended_labels,
    )
    plt.title(f'{title_prefix} – Confusion Matrix Across All States')
    plt.ylabel('Actual State')
    plt.xlabel('Predicted State')
    cm_path = os.path.join(output_dir, f"{title_prefix}_confusion_matrix_multiclass.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved confusion matrix to {cm_path}")

    # --- E. Single Micro-average ROC Curve & AUC ---
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

# --- 4. Generate Global and Per-Subject Reports ---
print("\n--- Generating Global Results ---")
generate_and_save_results(df, os.path.join(RESULTS_DIR, "Global"), "Global")

# --- Additional: Generate silo-specific global reports ---
SILO_DEFINITIONS = {
    "HeadPose": ['Forward Pose', 'Looking Left', 'Looking Right', 'Looking Up', 'Looking Down'],
    "GazeDirection": ['Gaze Left', 'Gaze Right', 'Gaze Center'],
    "Smile": ['Smile'],
    "Yawning": ['Yawning'],
    "HandRaise": ['Hand Raise'],
    "Drowsy": ['Drowsy'],
}

for silo_name, states_list in SILO_DEFINITIONS.items():
    silo_title = f"Global_{silo_name}"
    generate_and_save_results(df, os.path.join(RESULTS_DIR, "Global"), silo_title, states_subset=states_list)

print("\n--- Generating Per-Subject Results ---")
for subject_id, subject_df in df.groupby('subject_id'):
    print(f"- Processing {subject_id}...")
    generate_and_save_results(subject_df, os.path.join(RESULTS_DIR, subject_id), subject_id)

    # Also generate silo-specific reports per subject
    for silo_name, states_list in SILO_DEFINITIONS.items():
        silo_title = f"{subject_id}_{silo_name}"
        generate_and_save_results(subject_df, os.path.join(RESULTS_DIR, subject_id), silo_title, states_subset=states_list)

print("\nReport generation complete!")
