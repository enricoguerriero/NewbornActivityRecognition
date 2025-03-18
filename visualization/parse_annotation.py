import os
import re
import pandas as pd
from glob import glob

# Define input and output directories
ANNOTATIONS_DIR = "data/annotations"
OUTPUT_DIR = "visualizations/annotations_csv"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the dataset splits
splits = ["train", "validation", "test"]

# Regex pattern to capture: event type, start_frame, end_frame, duration_frames
pattern = re.compile(r"^(.*?)\s+(\d+)\s+(\d+)\s+(\d+)")

for split in splits:
    split_dir = os.path.join(ANNOTATIONS_DIR, split)
    annotation_files = glob(os.path.join(split_dir, "*.txt"))
    rows = []
    
    for file_path in annotation_files:
        # Use the file name (without extension) as the video_id.
        base_name = os.path.basename(file_path)
        video_id = os.path.splitext(base_name)[0]
        
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = pattern.match(line)
                if match:
                    event_type = match.group(1).strip()
                    start_frame = int(match.group(2))
                    end_frame = int(match.group(3))
                    duration = int(match.group(4))
                    rows.append({
                        "video_id": video_id,
                        "annotation_file": base_name,
                        "event_type": event_type,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "duration_frames": duration
                    })
                else:
                    print(f"Line did not match pattern in file {file_path}: {line}")
    
    # Create a DataFrame and write it to a CSV file
    df = pd.DataFrame(rows)
    output_csv = os.path.join(OUTPUT_DIR, f"{split}_annotations.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved parsed annotations for split '{split}' to {output_csv}")
