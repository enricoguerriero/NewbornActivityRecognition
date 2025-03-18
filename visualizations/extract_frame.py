import os
import cv2
import pandas as pd

# Constants and directory paths
FPS_DEFAULT = 30  # Default fps if video metadata is unavailable.
VIDEOS_DIR = "data/videos"
ANNOTATIONS_CSV_DIR = "visualizations/annotations_csv"
OUTPUT_FRAMES_DIR = "visualizations/frames"

# Create the output directory if needed
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

splits = ["train", "validation", "test"]

for split in splits:
    # Load the annotations CSV for the current split
    csv_path = os.path.join(ANNOTATIONS_CSV_DIR, f"{split}_annotations.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file for split {split} not found: {csv_path}")
        continue
    
    df = pd.read_csv(csv_path)
    
    # Create an output directory for the split
    split_output_dir = os.path.join(OUTPUT_FRAMES_DIR, split)
    os.makedirs(split_output_dir, exist_ok=True)
    
    # Process each video (grouped by video_id)
    video_ids = df["video_id"].unique()
    for video_id in video_ids:
        video_path = os.path.join(VIDEOS_DIR, split, video_id + ".mp4")
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            continue
        
        # Retrieve the FPS; fallback to default if unavailable.
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = FPS_DEFAULT
        
        # Create a folder for the current video
        video_output_dir = os.path.join(split_output_dir, video_id)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Get events for this video
        video_events = df[df["video_id"] == video_id]
        for idx, row in video_events.iterrows():
            # Here, we interpret 'start_frame' as the start time in ms
            start_time_ms = row["start_frame"]
            event_type = row["event_type"].replace(" ", "_")
            
            # Convert milliseconds to the corresponding frame index
            frame_index = int(start_time_ms / 1000.0 * fps)
            timestamp_sec = start_time_ms / 1000.0  # for optional logging
            
            # Set the video to the computed frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                # Save the frame as a JPG image
                output_filename = f"{event_type}_{frame_index}_frame.jpg"
                output_path = os.path.join(video_output_dir, output_filename)
                cv2.imwrite(output_path, frame)
                print(f"Saved frame for video {video_id} at time {timestamp_sec:.2f}s (frame {frame_index}): {output_path}")
            else:
                print(f"Failed to read frame at time {timestamp_sec:.2f}s (frame {frame_index}) in video {video_id}")
        
        cap.release()
