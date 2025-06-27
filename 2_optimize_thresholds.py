import pandas as pd
import numpy as np
import glob
import os
import optuna
from sklearn.metrics import accuracy_score
import json

# --- Path to your final, enriched data directory ---
PROCESSED_DATA_DIR = "enriched_processed_data" 

# --- Configuration for the dat cleaning process ---
TRIM_PERCENTAGE = 0.1  # Remove first and last 15% of frames for each state segment
IQR_MULTIPLIER = 3   # Standard value for outlier detection.

# --- Data Cleaning Function ---
def clean_data(df, all_csv_files):
    print("\n--- Starting Data Cleaning Process ---")
    df['video_file'] = ''
    start_index = 0
    for file in all_csv_files:
        try:
            sub_df_len = len(pd.read_csv(file))
            end_index = start_index + sub_df_len
            df.loc[start_index:end_index, 'video_file'] = os.path.basename(file)
            start_index = end_index
        except Exception as e:
            print(f"Warning: Could not process file {file} for video name mapping. Error: {e}")
        
    rows_before_cleaning = len(df)
    indices_to_drop = set()

    state_to_metric_map = {
        "Drowsy": "ear", "Yawning": "mar", "Gaze Left": "gaze_avg_ratio",
        "Gaze Right": "gaze_avg_ratio", "Gaze Center": "gaze_avg_ratio",
        "Looking Left": "pose_horiz_centered", "Looking Right": "pose_horiz_centered",
        "Hand Raise": "hand_raise_metric", "Forward Pose": "pose_forward_metric",
        "Emotion Happy": "mouth_curl_metric", "Emotion Sad": "mouth_curl_metric",
        "Emotion Surprise": "emotion_surprise_metric",
    }
    
    for (video, scenario), group in df.groupby(['video_file', 'scenario_ground_truth']):
        if scenario not in state_to_metric_map:
            continue
        metric = state_to_metric_map[scenario]
        n_frames = len(group)
        trim_count = int(n_frames * TRIM_PERCENTAGE)
        
        core_group = group.iloc[trim_count : n_frames - trim_count]
        indices_to_drop.update(group.head(trim_count).index)
        indices_to_drop.update(group.tail(trim_count).index)
        
        valid_metric_data = core_group[metric].dropna()
        if len(valid_metric_data) > 4:
            Q1, Q3 = valid_metric_data.quantile(0.25), valid_metric_data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower_bound, upper_bound = Q1 - (IQR_MULTIPLIER * IQR), Q3 + (IQR_MULTIPLIER * IQR)
                outlier_mask = (core_group[metric] < lower_bound) | (core_group[metric] > upper_bound)
                indices_to_drop.update(core_group[outlier_mask].index)
            
    df_cleaned = df.drop(index=list(indices_to_drop)).copy()
    print(f" -> Removed {rows_before_cleaning - len(df_cleaned)} frames (transition/outlier).")
    print(f" -> {len(df_cleaned)} high-quality frames remaining for optimization.")
    return df_cleaned

# --- 1. Load and Clean Data ---
all_csv_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "*.csv"))
if not all_csv_files:
    print(f"Error: No CSV files found in '{PROCESSED_DATA_DIR}'. Run the enrichment script first.")
    exit()
df_raw = pd.concat([pd.read_csv(f) for f in all_csv_files], ignore_index=True)
print(f"Loaded {len(df_raw)} total frames from {len(all_csv_files)} enriched data files.")
df = clean_data(df_raw, all_csv_files)

# ==============================================================================
#                 SILO-BASED OPTIMIZATION
# ==============================================================================
final_params = {}

# --- Silo 1: Drowsiness System ---
def objective_drowsy(trial):
    ear_thresh = trial.suggest_float("ear_thresh", 0.1, 0.4)
    drowsy_frames = trial.suggest_int("drowsy_frames", 5, 60)
    drowsy_pred = df['ear'].rolling(window=drowsy_frames, min_periods=1).mean() < ear_thresh
    mask = (df['scenario_ground_truth'] == "Drowsy") & df['ear'].notna()
    return accuracy_score(np.ones(mask.sum()), drowsy_pred[mask]) if mask.sum() > 0 else 0.0

print("\n--- Optimizing Drowsiness System ---")
study_drowsy = optuna.create_study(direction="maximize")
study_drowsy.optimize(objective_drowsy, n_trials=300)
final_params.update(study_drowsy.best_params)

# --- Silo 2: Yawn System ---
def objective_yawn(trial):
    mar_thresh = trial.suggest_float("mar_thresh", 0.4, 1.0)
    yawn_frames = trial.suggest_int("yawn_frames", 5, 40)
    yawn_pred = df['mar'].rolling(window=yawn_frames, min_periods=1).mean() > mar_thresh
    mask = (df['scenario_ground_truth'] == "Yawning") & df['mar'].notna()
    return accuracy_score(np.ones(mask.sum()), yawn_pred[mask]) if mask.sum() > 0 else 0.0

print("\n--- Optimizing Yawn System ---")
study_yawn = optuna.create_study(direction="maximize")
study_yawn.optimize(objective_yawn, n_trials=300)
final_params.update(study_yawn.best_params)

# --- Silo 3: Hand Raise System ---
def objective_hand(trial):
    hand_raise_thresh = trial.suggest_float("hand_raise_thresh", 0.2, 2.5)
    hand_pred = df['hand_raise_metric'] > hand_raise_thresh
    mask = (df['scenario_ground_truth'] == "Hand Raise") & df['hand_raise_metric'].notna()
    return accuracy_score(np.ones(mask.sum()), hand_pred[mask]) if mask.sum() > 0 else 0.0

print("\n--- Optimizing Hand Raise System ---")
study_hand = optuna.create_study(direction="maximize")
study_hand.optimize(objective_hand, n_trials=300)
final_params.update(study_hand.best_params)

# --- Silo 4: Gaze System ---
def objective_gaze(trial):
    gaze_left_thresh = trial.suggest_float("gaze_left_thresh", 0.2, 0.5)
    gaze_right_thresh = trial.suggest_float("gaze_right_thresh", 0.5, 0.8)
    gaze_left_pred = df['gaze_avg_ratio'] < gaze_left_thresh
    gaze_right_pred = df['gaze_avg_ratio'] > gaze_right_thresh
    gaze_center_pred = (df['gaze_avg_ratio'] >= gaze_left_thresh) & (df['gaze_avg_ratio'] <= gaze_right_thresh)
    
    y_true, y_pred = [], []
    for state, pred in [("Gaze Left", gaze_left_pred), ("Gaze Right", gaze_right_pred), ("Gaze Center", gaze_center_pred)]:
        mask = (df['scenario_ground_truth'] == state) & df['gaze_avg_ratio'].notna()
        if mask.sum() > 0:
            y_true.extend(np.ones(mask.sum())); y_pred.extend(pred[mask].astype(int))
    return accuracy_score(y_true, y_pred) if y_true else 0.0

print("\n--- Optimizing Gaze System ---")
study_gaze = optuna.create_study(direction="maximize")
study_gaze.optimize(objective_gaze, n_trials=300)
final_params.update(study_gaze.best_params)

# --- Silo 5: Head Pose System ---
def objective_pose(trial):
    # Head turns now use centred metric (-1 .. +1)
    pose_h_left_thresh = trial.suggest_float("pose_h_left_thresh", -1.0, -0.15)
    pose_h_right_thresh = trial.suggest_float("pose_h_right_thresh", 0.15, 1.0)
    pose_forward_thresh = trial.suggest_float("pose_forward_thresh", 0.05, 0.3)

    pose_left_pred = df['pose_horiz_centered'] < pose_h_left_thresh
    pose_right_pred = df['pose_horiz_centered'] > pose_h_right_thresh
    pose_forward_pred = df['pose_forward_metric'] < pose_forward_thresh
    # Note: We are not optimizing for Looking Up/Down as we don't have a good vertical metric yet.
    
    y_true, y_pred = [], []
    for state, pred in [("Looking Left", pose_left_pred), ("Looking Right", pose_right_pred), ("Forward Pose", pose_forward_pred)]:
        mask = (df['scenario_ground_truth'] == state) & (df['pose_horiz_centered'].notna() & df['pose_forward_metric'].notna())
        if mask.sum() > 0:
            y_true.extend(np.ones(mask.sum())); y_pred.extend(pred[mask].astype(int))
    return accuracy_score(y_true, y_pred) if y_true else 0.0

print("\n--- Optimizing Head Pose System ---")
study_pose = optuna.create_study(direction="maximize")
study_pose.optimize(objective_pose, n_trials=300)
final_params.update(study_pose.best_params)

# --- NEW Silo: Vertical Head Pose System (Looking Up / Down) ---

def objective_pose_vert(trial):
    # Pose vertical ratio: 0 (top) .. 1 (bottom) roughly
    pose_v_up_thresh = trial.suggest_float("pose_v_up_thresh", 0.1, 0.45)
    pose_v_down_thresh = trial.suggest_float("pose_v_down_thresh", 0.55, 0.9)

    pose_up_pred = df['pose_vert_ratio'] < pose_v_up_thresh
    pose_down_pred = df['pose_vert_ratio'] > pose_v_down_thresh

    y_true, y_pred = [], []
    for state, pred in [("Looking Up", pose_up_pred), ("Looking Down", pose_down_pred)]:
        mask = (df['scenario_ground_truth'] == state) & df['pose_vert_ratio'].notna()
        if mask.sum() > 0:
            y_true.extend(np.ones(mask.sum())); y_pred.extend(pred[mask].astype(int))
    return accuracy_score(y_true, y_pred) if y_true else 0.0

print("\n--- Optimizing Vertical Head Pose System ---")
study_pose_vert = optuna.create_study(direction="maximize")
study_pose_vert.optimize(objective_pose_vert, n_trials=300)
final_params.update(study_pose_vert.best_params)

# --- Silo 6: Smile (Happy) System ---
def objective_smile(trial):
    # Detect only smiles; anything else = non-smile.
    smile_thresh = trial.suggest_float("smile_thresh", -0.15, -0.001)

    pred = df['mouth_curl_metric'] < smile_thresh

    mask = df['scenario_ground_truth'].isin(["Emotion Happy", "Emotion Sad", "Emotion Surprise"]) & df['mouth_curl_metric'].notna()
    if mask.sum() == 0:
        return 0.0

    y_true = (df.loc[mask, 'scenario_ground_truth'] == "Emotion Happy").astype(int)
    y_pred = pred[mask].astype(int)

    return accuracy_score(y_true, y_pred)

print("\n--- Optimizing Smile Detection ---")
study_smile = optuna.create_study(direction="maximize")
study_smile.optimize(objective_smile, n_trials=300)
final_params.update(study_smile.best_params)


# ==============================================================================
#                      FINAL RESULTS PRESENTATION
# ==============================================================================
print("\n========================================================")
print("     Silo-Based Optimization Finished!")
print("========================================================")

print("\n--- Final Optimized Thresholds from Each Independent System ---")
# Sort for readability
for k, v in sorted(final_params.items()):
    print(f"  {k:<25}: {v:.4f}" if isinstance(v, float) else f"  {k:<25}: {v}")

# NEW: Save thresholds to JSON for downstream scripts
with open("optimized_thresholds.json", "w") as fp:
    json.dump(final_params, fp, indent=4)
    print("\nThresholds saved to 'optimized_thresholds.json'.")

print("\n--- Final Accuracy Breakdown by State (using best thresholds) ---")
# Re-run all predictions using the final combined set of optimal parameters
params = final_params
smile_pred = df['mouth_curl_metric'] < params['smile_thresh']
sad_pred = None  # deprecated
surprise_pred = None
pose_forward_pred = df['pose_forward_metric'] < params['pose_forward_thresh']
hand_raised_pred = df['hand_raise_metric'] > params['hand_raise_thresh']
drowsy_pred = df['ear'].rolling(window=params['drowsy_frames'], min_periods=1).mean() < params['ear_thresh']
yawn_pred = df['mar'].rolling(window=params['yawn_frames'], min_periods=1).mean() > params['mar_thresh']
gaze_left_pred = df['gaze_avg_ratio'] < params['gaze_left_thresh']
gaze_right_pred = df['gaze_avg_ratio'] > params['gaze_right_thresh']
gaze_center_pred = (df['gaze_avg_ratio'] >= params['gaze_left_thresh']) & (df['gaze_avg_ratio'] <= params['gaze_right_thresh'])
pose_left_pred = df['pose_horiz_centered'] < params['pose_h_left_thresh']
pose_right_pred = df['pose_horiz_centered'] > params['pose_h_right_thresh']
pose_up_pred = df['pose_vert_ratio'] < params['pose_v_up_thresh']
pose_down_pred = df['pose_vert_ratio'] > params['pose_v_down_thresh']

# Final evaluation maps
preds = {
    "Smile": smile_pred,
    "Drowsy": drowsy_pred, "Yawning": yawn_pred, "Gaze Left": gaze_left_pred,
    "Gaze Right": gaze_right_pred, "Gaze Center": gaze_center_pred, "Looking Left": pose_left_pred,
    "Looking Right": pose_right_pred, "Looking Up": pose_up_pred, "Looking Down": pose_down_pred,
    "Hand Raise": hand_raised_pred, "Forward Pose": pose_forward_pred,
}
validity = {
    "Smile": df['mouth_curl_metric'].notna(), "Drowsy": df['ear'].notna(), "Yawning": df['mar'].notna(),
    "Gaze Left": df['gaze_avg_ratio'].notna(), "Gaze Right": df['gaze_avg_ratio'].notna(), "Gaze Center": df['gaze_avg_ratio'].notna(),
    "Looking Left": df['pose_horiz_centered'].notna(), "Looking Right": df['pose_horiz_centered'].notna(),
    "Looking Up": df['pose_vert_ratio'].notna(), "Looking Down": df['pose_vert_ratio'].notna(),
    "Hand Raise": df['hand_raise_metric'].notna(), "Forward Pose": df['pose_forward_metric'].notna(),
}

# Loop and calculate final accuracies
alias = {"Smile": "Emotion Happy"}

for s, p in sorted(preds.items()):
    if s not in validity:
        continue
    gt_label = alias.get(s, s)
    mask = (df['scenario_ground_truth'] == gt_label) & validity[s]
    if mask.sum() > 0:
        y_true = np.ones(mask.sum())
        y_pred = p[mask].astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        print(f"  - {s:<20}: {accuracy:.2%}")
    else:
        print(f"  - {s:<20}: N/A (No valid frames found)")