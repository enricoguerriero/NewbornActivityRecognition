import os
import cv2
import pandas as pd

def extract_frames(annotations_csv_dir, videos_dir, output_frames_dir, splits=["train", "validation", "test"], fps_default=30):
    for split in splits:
        csv_path = os.path.join(annotations_csv_dir, f"{split}_annotations.csv")
        if not os.path.exists(csv_path):
            print(f"CSV file for split '{split}' not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        split_output_dir = os.path.join(output_frames_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        video_ids = df["video_id"].unique()
        for video_id in video_ids:
            video_path = os.path.join(videos_dir, split, video_id + ".mp4")
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                continue
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = fps_default
            video_output_dir = os.path.join(split_output_dir, video_id)
            os.makedirs(video_output_dir, exist_ok=True)
            video_events = df[df["video_id"] == video_id]
            for idx, row in video_events.iterrows():
                # Interpret start_ms as the start time in milliseconds
                start_time_ms = row["start_ms"]
                event_type = row["event_type"].replace(" ", "_")
                # Convert timestamp (ms) to frame index using fps:
                frame_index = int(start_time_ms / 1000.0 * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if ret:
                    output_filename = f"{event_type}_{frame_index}_frame.jpg"
                    output_path = os.path.join(video_output_dir, output_filename)
                    cv2.imwrite(output_path, frame)
                    print(f"Saved frame for video {video_id} at {start_time_ms/1000.0:.2f}s (frame {frame_index}): {output_path}")
                else:
                    print(f"Failed to read frame at {start_time_ms/1000.0:.2f}s (frame {frame_index}) in video {video_id}")
            cap.release()

if __name__ == "__main__":
    annotations_csv_dir = os.path.join("visualizations", "annotations_csv")
    videos_dir = os.path.join("data", "videos")
    output_frames_dir = os.path.join("visualizations", "frames")
    extract_frames(annotations_csv_dir, videos_dir, output_frames_dir)
