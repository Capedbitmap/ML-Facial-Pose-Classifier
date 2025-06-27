import cv2
import mediapipe as mp
import csv
import os
import glob
from scipy.spatial import distance as dist
import numpy as np

# --- MODIFIED: Paths are now set for the root project folder ---
VIDEO_SOURCE_DIR = "." # Look for videos in the current directory
PROCESSED_DATA_DIR = "processed_data"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- (All utility functions like eye_aspect_ratio, etc., remain the same) ---
def eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def mouth_aspect_ratio(top, bottom, left, right):
    vertical = dist.euclidean(top, bottom)
    horizontal = dist.euclidean(left, right)
    return vertical / horizontal if horizontal != 0 else 0.0

def get_pixel_coords(landmark, img_w, img_h):
    return int(landmark.x * img_w), int(landmark.y * img_h)

def get_eye_ratio(eye_corner1, eye_corner2, iris):
    eye_width = abs(eye_corner2[0] - eye_corner1[0])
    iris_offset = iris[0] - eye_corner1[0]
    return iris_offset / eye_width if eye_width != 0 else 0.5

# --- Setup Models (once) ---
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
pose = mp_pose.Pose()

# --- Scenarios and Timing Setup ---
scenarios = [
    "Looking Left", "Looking Right", "Looking Up", "Looking Down", "Forward Pose",
    "Yawning", "Emotion Happy", "Emotion Sad", "Emotion Surprise",
    "Hand Raise", "Drowsy", "Gaze Center", "Gaze Right", "Gaze Left"
]
SCENARIO_DURATION_SECONDS = 10
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
LEFT_IRIS_IDX = 468
RIGHT_IRIS_IDX = 473

# --- Main Loop to Process All Videos ---
video_files = glob.glob(os.path.join(VIDEO_SOURCE_DIR, "*.mp4"))
print(f"Found {len(video_files)} videos to process in the project folder.")

for video_path in video_files:
    video_name = os.path.basename(video_path)
    print(f"Processing: {video_name}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  -> Could not open video. Skipping.")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    output_csv_filename = video_name.replace('.mp4', '_data.csv')
    csv_path = os.path.join(PROCESSED_DATA_DIR, output_csv_filename)

    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "frame", "timestamp_ms", "scenario_ground_truth",
            "ear", "mar", "pose_horiz_ratio", "pose_vert_ratio", "gaze_avg_ratio", "hand_raised_flag"
        ])

        frame_counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            current_scenario_index = int(frame_counter / (fps * SCENARIO_DURATION_SECONDS))
            if current_scenario_index >= len(scenarios): break
            scenario_ground_truth = scenarios[current_scenario_index]
            
            img_h, img_w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            pose_results = pose.process(rgb_frame)
            
            ear_val, mar_val, pose_h_ratio, pose_v_ratio, gaze_ratio, hand_flag = 0.0, 0.0, 0.5, 0.5, 0.5, 0

            if results.multi_face_landmarks:
                lm_list = results.multi_face_landmarks[0].landmark
                pixel_landmarks = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in lm_list]
                
                left_eye_pts = [pixel_landmarks[i] for i in LEFT_EYE_IDX]
                right_eye_pts = [pixel_landmarks[i] for i in RIGHT_EYE_IDX]
                ear_val = (eye_aspect_ratio(left_eye_pts) + eye_aspect_ratio(right_eye_pts)) / 2.0
                mar_val = mouth_aspect_ratio(pixel_landmarks[13], pixel_landmarks[14], pixel_landmarks[78], pixel_landmarks[308])
                nose, face_top, face_bottom, face_left, face_right = pixel_landmarks[1], pixel_landmarks[10], pixel_landmarks[152], pixel_landmarks[234], pixel_landmarks[454]
                face_width = dist.euclidean(face_left, face_right)
                face_height = dist.euclidean(face_top, face_bottom)
                if face_width > 0: pose_h_ratio = (nose[0] - face_left[0]) / face_width
                if face_height > 0: pose_v_ratio = (nose[1] - face_top[1]) / face_height
                l_gaze = get_eye_ratio(get_pixel_coords(lm_list[33], img_w, img_h), get_pixel_coords(lm_list[133], img_w, img_h), get_pixel_coords(lm_list[LEFT_IRIS_IDX], img_w, img_h))
                r_gaze = get_eye_ratio(get_pixel_coords(lm_list[362], img_w, img_h), get_pixel_coords(lm_list[263], img_w, img_h), get_pixel_coords(lm_list[RIGHT_IRIS_IDX], img_w, img_h))
                gaze_ratio = (l_gaze + r_gaze) / 2.0

            if pose_results.pose_landmarks:
                pl = pose_results.pose_landmarks.landmark
                if (pl[mp_pose.PoseLandmark.LEFT_WRIST].y < pl[mp_pose.PoseLandmark.NOSE].y) or (pl[mp_pose.PoseLandmark.RIGHT_WRIST].y < pl[mp_pose.PoseLandmark.NOSE].y):
                    hand_flag = 1
            
            writer.writerow([frame_counter, cap.get(cv2.CAP_PROP_POS_MSEC), scenario_ground_truth, ear_val, mar_val, pose_h_ratio, pose_v_ratio, gaze_ratio, hand_flag])
            frame_counter += 1

    cap.release()
    print(f"  -> Finished. Data saved to '{csv_path}'")

face_mesh.close()
pose.close()
print("\nAll videos processed successfully!")