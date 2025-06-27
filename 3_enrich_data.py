import pandas as pd
import numpy as np
import glob
import os
import cv2
import mediapipe as mp
from scipy.spatial import distance as dist

# --- Define Your Project Paths ---
# Source videos are in the main project folder
VIDEO_SOURCE_DIR = "." 
# Source CSVs are in the 'processed_data' folder
PROCESSED_DATA_DIR = "processed_data"
# The new, separate directory for our output
ENRICHED_DATA_DIR = "enriched_processed_data"

# Create the output directory if it doesn't exist
os.makedirs(ENRICHED_DATA_DIR, exist_ok=True)

# --- Setup MediaPipe Pose Model (once for efficiency) ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

print("Starting script to enrich data with the 'hand_raise_metric'...")
print(f"Source CSVs will be read from: '{PROCESSED_DATA_DIR}'")
print(f"Enriched CSVs will be saved to: '{ENRICHED_DATA_DIR}'")

# --- Find all the video files ---
video_files = glob.glob(os.path.join(VIDEO_SOURCE_DIR, "*.mp4"))
if not video_files:
    print(f"Error: No .mp4 video files found in the main project directory ('{VIDEO_SOURCE_DIR}').")
    exit()

# --- Loop through each video to process it ---
for video_path in video_files:
    video_name = os.path.basename(video_path)
    
    # Construct the path to the corresponding source CSV file in 'processed_data'
    source_csv_name = video_name.replace('.mp4', '_data.csv')
    source_csv_path = os.path.join(PROCESSED_DATA_DIR, source_csv_name)

    if not os.path.exists(source_csv_path):
        print(f"  -> WARNING: Source data file not found at '{source_csv_path}'. Skipping this video.")
        continue

    print(f"\nProcessing '{video_name}' and reading '{source_csv_path}'...")

    # Load the existing data that you have already labeled
    df = pd.read_csv(source_csv_path)
    
    # Initialize the new column that will hold our better metric
    df['hand_raise_metric'] = None

    # Open the video file to re-analyze poses
    cap = cv2.VideoCapture(video_path)
    frame_counter = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hand_raise_metric_val = None 
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)

        if pose_results.pose_landmarks:
            pl = pose_results.pose_landmarks.landmark
            
            # Check visibility of key landmarks for a high-quality calculation
            if (pl[mp_pose.PoseLandmark.NOSE].visibility > 0.5 and
                pl[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.5 and
                pl[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.5 and
                (pl[mp_pose.PoseLandmark.LEFT_WRIST].visibility > 0.5 or 
                 pl[mp_pose.PoseLandmark.RIGHT_WRIST].visibility > 0.5)):

                nose_y = pl[mp_pose.PoseLandmark.NOSE].y
                left_shoulder_y = pl[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                right_shoulder_y = pl[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                
                shoulder_y = min(left_shoulder_y, right_shoulder_y)
                face_height_ref = shoulder_y - nose_y
                
                if face_height_ref > 0.01: # Use a small threshold to avoid noise
                    left_hand_height = shoulder_y - pl[mp_pose.PoseLandmark.LEFT_WRIST].y
                    right_hand_height = shoulder_y - pl[mp_pose.PoseLandmark.RIGHT_WRIST].y
                    
                    hand_raise_metric_val = max(left_hand_height, right_hand_height) / face_height_ref
        
        # Add the calculated metric to the correct row in our loaded DataFrame
        if frame_counter < len(df):
             df.at[frame_counter, 'hand_raise_metric'] = hand_raise_metric_val

        frame_counter += 1

    cap.release()
    
    # --- Cleanup and Save to the NEW location ---
    if 'hand_raised_flag' in df.columns:
        df = df.drop(columns=['hand_raised_flag'])
        print(f"  -> Old 'hand_raised_flag' column removed.")

    # Construct the NEW, enriched filename and path
    enriched_csv_name = video_name.replace('.mp4', '_data_enriched.csv')
    enriched_csv_path = os.path.join(ENRICHED_DATA_DIR, enriched_csv_name)

    # Save the updated DataFrame to the new file in the new directory
    df.to_csv(enriched_csv_path, index=False, float_format='%.6f')
    print(f"  -> Successfully created enriched file at '{enriched_csv_path}'")

pose.close()
print("\nAll data files have been enriched successfully!")