import os
import pandas as pd
import plotly.express as px

def generate_timelines(annotations_csv_dir, output_timeline_dir, splits=["train", "validation", "test"]):
    os.makedirs(output_timeline_dir, exist_ok=True)
    for split in splits:
        csv_path = os.path.join(annotations_csv_dir, f"{split}_annotations.csv")
        if not os.path.exists(csv_path):
            print(f"CSV file for split '{split}' not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        # Convert timestamps from ms to sec
        df["start_sec"] = df["start_ms"] / 1000.0
        df["end_sec"] = df["end_ms"] / 1000.0

        # Generate a timeline (Gantt chart) for each video
        video_ids = df["video_id"].unique()
        for video in video_ids:
            subset_video = df[df["video_id"] == video].copy()
            subset_video["Task"] = subset_video["event_type"]
            subset_video["Start"] = subset_video["start_sec"]
            subset_video["Finish"] = subset_video["end_sec"]
            fig = px.timeline(subset_video, x_start="Start", x_end="Finish", y="Task", color="event_type",
                              title=f"Event Timeline for Video {video} ({split} split)")
            fig.update_yaxes(autorange="reversed")
            output_file = os.path.join(output_timeline_dir, f"timeline_{video}_{split}.html")
            fig.write_html(output_file)
            print(f"Saved timeline for video {video} in {split} at {output_file}")

if __name__ == "__main__":
    annotations_csv_dir = os.path.join("visualizations", "annotations_csv")
    output_timeline_dir = os.path.join("visualizations", "timelines")
    generate_timelines(annotations_csv_dir, output_timeline_dir)
