import pandas as pd
import numpy as np
import glob
import os
import cv2
import mediapipe as mp
from scipy.spatial import distance as dist

# --- Utility Functions ---
def eye_aspect_ratio(eye_points):
    """Calculates the Eye Aspect Ratio for a single eye."""
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def mouth_aspect_ratio(top, bottom, left, right):
    """Calculates the Mouth Aspect Ratio."""
    vertical = dist.euclidean(top, bottom)
    horizontal = dist.euclidean(left, right)
    return vertical / horizontal if horizontal != 0 else 0.0

# --- Define Project Paths ---
VIDEO_SOURCE_DIR = "videos"
# We will READ from your original processed data
SOURCE_DATA_DIR = "processed_data"
# We will WRITE the new files to a separate, new directory
ENRICHED_DATA_DIR = "enriched_processed_data"

# Create the new directory if it doesn't already exist
os.makedirs(ENRICHED_DATA_DIR, exist_ok=True)

# --- Setup MediaPipe Models ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

print("Starting data enrichment process...")
print(f"Reading source CSVs from: '{SOURCE_DATA_DIR}'")
print(f"Saving enriched CSVs to: '{ENRICHED_DATA_DIR}'")

# --- Main Loop ---
for video_path in glob.glob(os.path.join(VIDEO_SOURCE_DIR, "*.mp4")):
    video_name = os.path.basename(video_path)
    source_csv_path = os.path.join(SOURCE_DATA_DIR, video_name.replace('.mp4', '_data.csv'))

    if not os.path.exists(source_csv_path):
        print(f"  -> WARNING: Source CSV not found at '{source_csv_path}'. Skipping this video.")
        continue

    print(f"\nProcessing '{video_name}'...")
    df = pd.read_csv(source_csv_path)

    # Initialize all the new columns we want to add
    df['hand_raise_metric'] = None
    df['pose_forward_metric'] = None
    df['mouth_curl_metric'] = None # New unified metric for happy/sad
    df['emotion_surprise_metric'] = None
    df['pose_horiz_centered'] = None  # Normalised horizontal head position (-1 = left, +1 = right)

    cap = cv2.VideoCapture(video_path)
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Reset metric values for the current frame
        hand_raise_val, pose_fwd_val, mouth_curl_val, surprise_val = None, None, None, None

        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_results = face_mesh.process(rgb_frame)
        pose_results = pose.process(rgb_frame)

        if face_results.multi_face_landmarks:
            lm = face_results.multi_face_landmarks[0].landmark
            pixel_landmarks = [(int(p.x * img_w), int(p.y * img_h)) for p in lm]
            
            face_top = pixel_landmarks[10]; face_bottom = pixel_landmarks[152]
            face_left = pixel_landmarks[234]; face_right = pixel_landmarks[454]
            face_height = dist.euclidean(face_top, face_bottom)
            face_width  = dist.euclidean(face_left, face_right)
            face_diagonal = np.sqrt(face_height**2 + face_width**2)

            # Forward-leaning metric (unchanged logic)
            if face_diagonal > 0:
                nose_pt = pixel_landmarks[1]
                face_center_x = (face_left[0] + face_right[0]) / 2
                face_center_y = (face_top[1] + face_bottom[1]) / 2
                nose_dist_from_center = dist.euclidean(nose_pt, (face_center_x, face_center_y))
                pose_fwd_val = nose_dist_from_center / face_diagonal

            # NEW: Normalised horizontal head position
            pose_horiz_centered_val = None
            if face_width > 0:
                face_center_x = (face_left[0] + face_right[0]) / 2
                nose_x = pixel_landmarks[1][0]
                pose_horiz_centered_val = (nose_x - face_center_x) / (face_width / 2)  # -1 .. +1 ideally

            if face_height > 0:
                top_lip = pixel_landmarks[13]
                bottom_lip = pixel_landmarks[14]
                left_corner = pixel_landmarks[61]; right_corner = pixel_landmarks[291]
                avg_corner_y = (left_corner[1] + right_corner[1]) / 2

                # Re-defined curl: negative when corners raised (smile), positive when lowered (sad)
                mouth_center_y = (top_lip[1] + bottom_lip[1]) / 2
                mouth_curl_val = (avg_corner_y - mouth_center_y) / face_height

                mar = mouth_aspect_ratio(top_lip, bottom_lip, pixel_landmarks[78], pixel_landmarks[308])
                left_eye_pts = [pixel_landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
                right_eye_pts = [pixel_landmarks[i] for i in [263, 387, 385, 362, 380, 373]]
                ear = (eye_aspect_ratio(left_eye_pts) + eye_aspect_ratio(right_eye_pts)) / 2.0
                surprise_val = mar + (0.5 * ear)

        if pose_results.pose_landmarks:
            pl = pose_results.pose_landmarks.landmark
            if (pl[mp_pose.PoseLandmark.NOSE].visibility > 0.5 and pl[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.5):
                nose_y = pl[mp_pose.PoseLandmark.NOSE].y
                shoulder_y = min(pl[mp_pose.PoseLandmark.LEFT_SHOULDER].y, pl[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
                face_height_ref = shoulder_y - nose_y
                if face_height_ref > 0.01:
                    left_hand_h = shoulder_y - pl[mp_pose.PoseLandmark.LEFT_WRIST].y
                    right_hand_h = shoulder_y - pl[mp_pose.PoseLandmark.RIGHT_WRIST].y
                    hand_raise_val = max(left_hand_h, right_hand_h) / face_height_ref

        if frame_counter < len(df):
            df.at[frame_counter, 'hand_raise_metric'] = hand_raise_val
            df.at[frame_counter, 'pose_forward_metric'] = pose_fwd_val
            df.at[frame_counter, 'pose_horiz_centered'] = pose_horiz_centered_val
            df.at[frame_counter, 'mouth_curl_metric'] = mouth_curl_val
            df.at[frame_counter, 'emotion_surprise_metric'] = surprise_val
        
        frame_counter += 1

    cap.release()
    
    # Clean up old columns before saving
    cols_to_drop = ['hand_raised_flag', 'emotion_happy_metric', 'emotion_sad_metric']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    enriched_csv_name = video_name.replace('.mp4', '_data_enriched.csv')
    enriched_csv_path = os.path.join(ENRICHED_DATA_DIR, enriched_csv_name)
    df.to_csv(enriched_csv_path, index=False, float_format='%.6f')
    print(f"  -> Successfully created enriched file: '{enriched_csv_path}'")

pose.close()
face_mesh.close()
print("\nData enrichment complete for all files!")