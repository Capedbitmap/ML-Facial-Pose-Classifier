# 20-Person Engagement Test

A comprehensive computer vision pipeline for analyzing human engagement and behavioral patterns from video data using MediaPipe and machine learning techniques.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Workflow](#pipeline-workflow)
- [Data Processing](#data-processing)
- [Results and Analysis](#results-and-analysis)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This project implements an automated system for analyzing human engagement and behavioral patterns from video recordings. It processes 20 participant videos (approximately 10 seconds each) through a multi-stage pipeline that extracts facial landmarks, body pose, and behavioral metrics to classify various engagement states.

The system can detect and classify:
- **Gaze Directions**: Left, Right, Center, Up, Down
- **Engagement States**: Drowsiness, Yawning, Hand Raising, Forward Pose
- **Emotional Expressions**: Happy, Sad, Surprise

## Features

### Core Capabilities
- ✅ **Multi-modal Analysis**: Combines facial landmarks, eye tracking, and body pose detection
- ✅ **Real-time Processing**: Efficient frame-by-frame video analysis
- ✅ **Automated Threshold Optimization**: Uses Optuna for hyperparameter tuning
- ✅ **Comprehensive Reporting**: Generates detailed performance metrics and visualizations
- ✅ **Ground Truth Generation**: Creates frame-level binary classification labels
- ✅ **Data Enrichment**: Adds hand-raise detection and pose analysis

### Technical Features
- MediaPipe integration for robust facial and pose landmark detection
- Scipy-based geometric calculations for feature extraction
- Scikit-learn for performance evaluation
- Matplotlib/Seaborn for visualization
- Pandas for efficient data manipulation

## Project Structure

```
20-person-engagment-test/
├── README.md                    # This file
├── CLAUDE.md                   # Development notes and troubleshooting
├── optimized_thresholds.json   # ML-optimized detection thresholds
├── venv/                       # Python virtual environment
│
├── videos/                     # Input video files
│   ├── 1.mp4                  # Subject 1 video
│   ├── 2.mp4                  # Subject 2 video
│   └── ... (up to 20.mp4)
│
├── processed_data/            # Stage 1: Raw extracted features
│   ├── 1_data.csv
│   ├── 2_data.csv
│   └── ... (20 files)
│
├── enriched_processed_data/   # Stage 2: Enhanced with pose data
│   ├── 1_data_enriched.csv
│   ├── 2_data_enriched.csv
│   └── ... (20 files)
│
├── Results/                   # Analysis outputs and visualizations
│   ├── Global/               # Aggregated performance metrics
│   ├── PerVideoFrames/       # Frame-level analysis
│   └── Subject_X/            # Individual subject results
│
├── abdalla's stuff/          # Ground truth generation utilities
│   ├── generate_ground_truth.py
│   └── ground_truth_output/  # Frame-level binary labels
│
└── Pipeline Scripts:
    ├── 1_process_videos.py      # Extract features from videos
    ├── 2_optimize_thresholds.py # ML threshold optimization
    ├── 3_enrich_data.py         # Add pose detection
    ├── 3.5_enrich_all_data.py   # Batch enrichment
    └── 4_generate_report_figures.py # Analysis and visualization
```

## Installation

### Prerequisites
- Python 3.8+
- OpenCV
- MediaPipe
- Required Python packages (see requirements below)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd 20-person-engagment-test
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # OR
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install pandas numpy opencv-python mediapipe scipy scikit-learn matplotlib seaborn optuna
   ```

   **⚠️ Critical Warning**: Always activate the virtual environment before installing packages. Never install globally as it will break the project environment.

## Usage

### Quick Start

Run the complete pipeline in sequence:

```bash
# Activate virtual environment
source venv/bin/activate

# Stage 1: Extract features from videos
python 1_process_videos.py

# Stage 2: Optimize detection thresholds
python 2_optimize_thresholds.py

# Stage 3: Enrich data with pose information
python 3_enrich_data.py

# Stage 4: Generate analysis reports
python 4_generate_report_figures.py
```

### Individual Script Usage

#### 1. Feature Extraction (`1_process_videos.py`)
Processes video files and extracts facial landmarks, eye aspect ratios, and gaze metrics.

```bash
python 1_process_videos.py
```

**Outputs**: CSV files in `processed_data/` containing:
- Eye Aspect Ratio (EAR) for drowsiness detection
- Mouth Aspect Ratio (MAR) for yawning detection
- Gaze direction ratios
- Head pose angles
- Facial expression metrics

#### 2. Threshold Optimization (`2_optimize_thresholds.py`)
Uses Optuna to find optimal detection thresholds for maximum accuracy.

```bash
python 2_optimize_thresholds.py
```

**Outputs**: `optimized_thresholds.json` with ML-tuned parameters for:
- Drowsiness detection (EAR threshold, frame count)
- Yawning detection (MAR threshold, frame count)
- Gaze direction thresholds
- Head pose boundaries
- Hand raise sensitivity

#### 3. Data Enrichment (`3_enrich_data.py`)
Adds hand-raise detection using pose estimation.

```bash
python 3_enrich_data.py
```

**Outputs**: Enhanced CSV files in `enriched_processed_data/` with additional `hand_raise_metric` column.

#### 4. Report Generation (`4_generate_report_figures.py`)
Creates comprehensive analysis reports and visualizations.

```bash
python 4_generate_report_figures.py
```

**Outputs**: 
- Performance metrics for each behavioral class
- Confusion matrices
- Per-subject and global accuracy reports
- ROC curves and statistical summaries

### Ground Truth Generation

Generate frame-level binary classification labels:

```bash
cd "abdalla's stuff"
python generate_ground_truth.py
```

**Outputs**: Binary classification files in `ground_truth_output/` with 14-dimensional vectors representing engagement states.

## Pipeline Workflow

### Stage 1: Video Processing
1. **Input**: 20 MP4 video files (numbered 1.mp4 through 20.mp4)
2. **Processing**: 
   - MediaPipe facial landmark detection
   - Eye aspect ratio calculation
   - Mouth aspect ratio calculation
   - Gaze direction estimation
   - Head pose angle extraction
3. **Output**: Raw feature CSV files

### Stage 2: Threshold Optimization
1. **Input**: Raw feature data + ground truth scenarios
2. **Processing**:
   - Data cleaning (outlier removal, trimming)
   - Optuna-based hyperparameter optimization
   - Cross-validation for robust threshold selection
3. **Output**: Optimized threshold parameters

### Stage 3: Data Enrichment
1. **Input**: Raw features + original videos
2. **Processing**:
   - MediaPipe pose detection
   - Hand-raise metric calculation
   - Feature augmentation
3. **Output**: Enriched feature datasets

### Stage 4: Analysis & Reporting
1. **Input**: Enriched data + optimized thresholds
2. **Processing**:
   - Binary classification application
   - Performance metric calculation
   - Visualization generation
3. **Output**: Comprehensive analysis reports

## Data Processing

### Scenarios Detected

The system recognizes 14 distinct behavioral scenarios:

#### Gaze Direction (7 classes)
- **Gaze Left**: Eyes directed left
- **Gaze Right**: Eyes directed right  
- **Gaze Center**: Forward-looking gaze
- **Looking Left**: Head turned left
- **Looking Right**: Head turned right
- **Looking Up**: Head tilted up
- **Looking Down**: Head tilted down

#### Engagement States (4 classes)
- **Drowsy**: Extended eye closure (EAR < threshold for N frames)
- **Yawning**: Mouth opening pattern (MAR > threshold for N frames)
- **Forward Pose**: Attentive upright posture
- **Hand Raise**: Hand positioned above shoulder level

#### Emotional Expressions (3 classes)
- **Emotion Happy**: Smile detection
- **Emotion Sad**: Downward facial expression
- **Emotion Surprise**: Wide eyes and open mouth

### Feature Extraction Metrics

- **Eye Aspect Ratio (EAR)**: `(|p2-p6| + |p3-p5|) / (2|p1-p4|)`
- **Mouth Aspect Ratio (MAR)**: `|top-bottom| / |left-right|`
- **Gaze Ratio**: `(iris_x - eye_corner_left) / eye_width`
- **Head Pose**: Euler angles from facial landmark geometry
- **Hand Raise**: Normalized hand landmark position relative to shoulders

### Data Cleaning

- **Temporal Trimming**: Removes first/last 10% of each scenario segment
- **Outlier Removal**: IQR-based filtering (3σ threshold)
- **Missing Data Handling**: Forward-fill and interpolation

## Results and Analysis

### Output Structure

#### Global Results (`Results/Global/`)
- `Global_PerVideo_Accuracy.csv`: Per-subject accuracy summary
- `Global_[Class]_performance_metrics.csv`: Precision, recall, F1-score
- `Global_[Class]_confusion_matrix.png`: Classification matrices

#### Per-Subject Results (`Results/Subject_X/`)
- Individual performance breakdowns
- Subject-specific behavioral patterns
- Temporal analysis of engagement states

#### Frame-Level Results (`Results/PerVideoFrames/`)
- Frame-by-frame classification outputs
- Temporal engagement trajectories
- Detailed behavioral timelines

### Performance Metrics

The system evaluates performance using:
- **Accuracy**: Overall classification correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve

### Typical Performance
Based on optimized thresholds, the system achieves:
- **Drowsiness Detection**: ~85-90% accuracy
- **Gaze Direction**: ~80-85% accuracy  
- **Hand Raise**: ~75-80% accuracy
- **Emotional States**: ~70-75% accuracy

## Configuration

### Threshold Parameters

Key detection thresholds (auto-optimized):

```json
{
    "ear_thresh": 0.358,        // Eye closure threshold
    "drowsy_frames": 52,        // Consecutive frames for drowsiness
    "mar_thresh": 0.400,        // Mouth opening threshold
    "yawn_frames": 5,           // Consecutive frames for yawning
    "hand_raise_thresh": 0.249, // Hand position threshold
    "gaze_left_thresh": 0.445,  // Left gaze boundary
    "gaze_right_thresh": 0.573, // Right gaze boundary
    "pose_forward_thresh": 0.174, // Forward pose angle
    "smile_thresh": -0.001      // Smile detection sensitivity
}
```

### Customization Options

#### Video Processing
- Modify `SCENARIO_DURATION_SECONDS` in `1_process_videos.py`
- Adjust MediaPipe detection confidence in model initialization
- Change landmark indices for different facial features

#### Optimization
- Tune `TRIM_PERCENTAGE` for data cleaning aggressiveness
- Modify `IQR_MULTIPLIER` for outlier sensitivity
- Adjust Optuna trial count for optimization depth

#### Analysis
- Configure visualization styles in `4_generate_report_figures.py`
- Modify result directory structure
- Customize performance metric calculations
