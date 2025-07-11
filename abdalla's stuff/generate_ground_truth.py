#!/usr/bin/env python3

import pandas as pd
import os
from pathlib import Path

def create_class_mapping():
    """
    Create mapping of ground truth labels to binary vector positions.
    Based on the 14 unique classes found in the enriched data.
    """
    classes = [
        # Gaze directions (0-6)
        'Gaze Left',
        'Gaze Right', 
        'Gaze Center',
        'Looking Left',
        'Looking Right',
        'Looking Up',
        'Looking Down',
        # Engagement/Attention (7-10)
        'Drowsy',
        'Yawning',
        'Forward Pose',
        'Hand Raise',
        # Emotions (11-13)
        'Emotion Happy',
        'Emotion Sad',
        'Emotion Surprise'
    ]
    
    return {label: idx for idx, label in enumerate(classes)}

def label_to_binary(label, class_mapping, num_classes):
    """
    Convert a ground truth label to binary vector.
    """
    binary_vector = ['0'] * num_classes
    if label in class_mapping:
        binary_vector[class_mapping[label]] = '1'
    return binary_vector

def process_video_file(csv_path, class_mapping, num_classes):
    """
    Process a single enriched CSV file and generate frame-level ground truth CSV.
    """
    try:
        df = pd.read_csv(csv_path)
        
        ground_truth_data = []
        
        for _, row in df.iterrows():
            frame_num = int(row['frame'])
            ground_truth_label = row['scenario_ground_truth']
            
            # Convert to binary format
            binary_vector = label_to_binary(ground_truth_label, class_mapping, num_classes)
            binary_string = ''.join(binary_vector)
            
            # Create CSV row
            ground_truth_data.append({
                'frame': f"{frame_num:05d}",
                'classes': f"[{binary_string}]"
            })
        
        return ground_truth_data
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return []

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    enriched_data_dir = base_dir / "enriched_processed_data"
    output_dir = Path(__file__).parent / "ground_truth_output"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Setup class mapping
    class_mapping = create_class_mapping()
    num_classes = len(class_mapping)
    
    print(f"Class mapping ({num_classes} classes):")
    for label, idx in class_mapping.items():
        print(f"  {idx:2d}: {label}")
    print()
    
    # Process all 20 enriched data files
    for video_num in range(1, 21):
        csv_filename = f"{video_num}_data_enriched.csv"
        csv_path = enriched_data_dir / csv_filename
        
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping...")
            continue
            
        print(f"Processing video {video_num}...")
        
        # Generate ground truth for this video
        ground_truth_data = process_video_file(csv_path, class_mapping, num_classes)
        
        if ground_truth_data:
            # Save to CSV file
            output_filename = f"video_{video_num}_ground_truth.csv"
            output_path = output_dir / output_filename
            
            ground_truth_df = pd.DataFrame(ground_truth_data)
            ground_truth_df.to_csv(output_path, index=False)
            
            print(f"  Generated {len(ground_truth_data)} frames -> {output_filename}")
        else:
            print(f"  Failed to process video {video_num}")
    
    print(f"\nGround truth files saved to: {output_dir}")
    print(f"Binary vector format: [{num_classes} bits]")
    
    # Print class reference
    print("\nClass Reference:")
    for label, idx in sorted(class_mapping.items(), key=lambda x: x[1]):
        print(f"Position {idx:2d}: {label}")

if __name__ == "__main__":
    main()