import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# --- Configuration ---
DATA_DIR = "enriched_processed_data"
RESULTS_DIR = "Results"

BEST_THRESHOLDS = {
    'hand_raise_thresh': 0.2481,
    'ear_thresh': 0.3944,
    'drowsy_frames': 5,
    'mar_thresh': 0.4017,
    'yawn_frames': 5,
    'gaze_left_thresh': 0.4320,
    'gaze_right_thresh': 0.5430,
    'pose_h_left_thresh': 0.3806,
    'pose_h_right_thresh': 0.5882
}

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
def generate_and_save_results(data, output_dir, title_prefix):
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
    predictions['Gaze Left'] = data['gaze_avg_ratio'] < BEST_THRESHOLDS['gaze_left_thresh']
    predictions['Gaze Right'] = data['gaze_avg_ratio'] > BEST_THRESHOLDS['gaze_right_thresh']
    predictions['Looking Left'] = data['pose_horiz_ratio'] < BEST_THRESHOLDS['pose_h_left_thresh']
    predictions['Looking Right'] = data['pose_horiz_ratio'] > BEST_THRESHOLDS['pose_h_right_thresh']
    
    predictions_df = pd.DataFrame(predictions)
    
    all_possible_states = list(predictions.keys())

    # --- B. Generate Classification Report (Precision/Recall/F1) ---
    report_data = []
    for state in all_possible_states:
        state_df = data[data['scenario_ground_truth'] == state].copy()
        if state_df.empty: continue
            
        y_true = np.ones(len(state_df))
        y_pred = predictions_df.loc[state_df.index, state].astype(int)
        
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        positive_class_stats = report.get('1', {})
        p, r, f1, s = [positive_class_stats.get(k, 0.0) for k in ['precision', 'recall', 'f1-score', 'support']]
        s = len(state_df) # Ensure support is correct
        report_data.append({'State': state, 'Precision': p, 'Recall': r, 'F1-Score': f1, 'Support': s})

    if not report_data:
        print(f"  -> No valid data to generate a report for {title_prefix}. Skipping.")
        return

    report_df = pd.DataFrame(report_data).set_index('State')
    report_path = os.path.join(output_dir, f"{title_prefix}_classification_report.csv")
    report_df.to_csv(report_path, float_format='%.3f')
    print(f"  -> Saved classification report to {report_path}")
    
    # --- C. Generate Binary Confusion Matrices for Each State ---
    cm_dir = os.path.join(output_dir, "Confusion_Matrices")
    os.makedirs(cm_dir, exist_ok=True)
    for state in all_possible_states:
        y_true = (data['scenario_ground_truth'] == state).astype(int)
        y_pred = predictions_df[state].astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Ensure the matrix is always 2x2, even if predictions are one-sided
        if cm.shape == (1, 1):
            # This means only one class was present in both y_true and y_pred
            # Figure out if it was the '0' or '1' class
            if y_true.unique()[0] == 0: # Only negatives
                cm = np.array([[cm[0][0], 0], [0, 0]]) # TN
            else: # Only positives
                cm = np.array([[0, 0], [0, cm[0][0]]]) # TP
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[f'Not {state}', state], 
                    yticklabels=[f'Not {state}', state])
        plt.title(f'{title_prefix} - Confusion Matrix for "{state}"')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        cm_path = os.path.join(cm_dir, f"{title_prefix}_cm_{state}.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
    print(f"  -> Saved all 2x2 confusion matrices to {cm_dir}")

    # --- D. Generate Meaningful ROC Curves ---
    score_map = {
        'Drowsy': (1 - data['ear']), 'Yawning': data['mar'], 'Hand Raise': data['hand_raise_metric'],
        'Gaze Left': (1 - data['gaze_avg_ratio']), 'Looking Left': (1 - data['pose_horiz_ratio']),
        'Gaze Right': data['gaze_avg_ratio'], 'Looking Right': data['pose_horiz_ratio']
    }
    plt.figure(figsize=(12, 10))
    for state, y_score in score_map.items():
        y_true = (data['scenario_ground_truth'] == state).astype(int)
        valid_indices = y_score.notna()
        y_true_filtered, y_score_filtered = y_true[valid_indices], y_score[valid_indices]
        if len(y_true_filtered.unique()) < 2: continue

        fpr, tpr, _ = roc_curve(y_true_filtered, y_score_filtered)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC for {state} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'{title_prefix} - One-vs-Rest ROC Curves'); plt.legend(loc="lower right")
    roc_path = os.path.join(output_dir, f"{title_prefix}_roc_curves.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved ROC curves to {roc_path}")

# --- 4. Generate Global and Per-Subject Reports ---
print("\n--- Generating Global Results ---")
generate_and_save_results(df, os.path.join(RESULTS_DIR, "Global"), "Global")

print("\n--- Generating Per-Subject Results ---")
for subject_id, subject_df in df.groupby('subject_id'):
    print(f"- Processing {subject_id}...")
    generate_and_save_results(subject_df, os.path.join(RESULTS_DIR, subject_id), subject_id)

print("\nReport generation complete!")
