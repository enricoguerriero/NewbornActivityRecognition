import os
import re
import pandas as pd
from glob import glob

def parse_annotations(data_dir, output_csv_dir, splits=["train", "validation", "test"]):
    os.makedirs(output_csv_dir, exist_ok=True)
    # This regex captures: event type, start time (ms), end time (ms), duration (ms)
    pattern = re.compile(r"^(.*?)\s+(\d+)\s+(\d+)\s+(\d+)")
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        annotation_files = glob(os.path.join(split_dir, "*.txt"))
        rows = []
        for file_path in annotation_files:
            base_name = os.path.basename(file_path)
            # Use the file name (without extension) as video_id.
            video_id = os.path.splitext(base_name)[0]
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    match = pattern.match(line)
                    if match:
                        event_type = match.group(1).strip()
                        start_time_ms = int(match.group(2))
                        end_time_ms = int(match.group(3))
                        duration_ms = int(match.group(4))
                        rows.append({
                            "video_id": video_id,
                            "annotation_file": base_name,
                            "event_type": event_type,
                            "start_ms": start_time_ms,
                            "end_ms": end_time_ms,
                            "duration_ms": duration_ms
                        })
                    else:
                        print(f"Line did not match pattern in file {file_path}: {line}")
        df = pd.DataFrame(rows)
        output_csv = os.path.join(output_csv_dir, f"{split}_annotations.csv")
        df.to_csv(output_csv, index=False)
        print(f"Saved parsed annotations for split '{split}' to {output_csv}")

if __name__ == "__main__":
    # data_dir: folder with annotation files (e.g., data/annotations)
    data_dir = os.path.join("data", "annotations")
    # output_csv_dir: where to store CSV files
    output_csv_dir = os.path.join("visualizations", "annotations_csv")
    parse_annotations(data_dir, output_csv_dir)
