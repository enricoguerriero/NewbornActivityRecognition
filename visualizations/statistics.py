import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_statistics(annotations_csv_dir, output_stats_dir, splits=["train", "validation", "test"]):
    os.makedirs(output_stats_dir, exist_ok=True)
    for split in splits:
        csv_path = os.path.join(annotations_csv_dir, f"{split}_annotations.csv")
        if not os.path.exists(csv_path):
            print(f"CSV file for split '{split}' not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        # Convert time values from milliseconds to seconds
        df["start_sec"] = df["start_ms"] / 1000.0
        df["duration_sec"] = df["duration_ms"] / 1000.0

        split_stats_dir = os.path.join(output_stats_dir, split)
        os.makedirs(split_stats_dir, exist_ok=True)

        # 1. Total Duration per Event Type
        total_duration = df.groupby("event_type")["duration_sec"].sum().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        total_duration.plot(kind="bar")
        plt.title(f"Total Duration per Event Type in {split.capitalize()} Split (sec)")
        plt.xlabel("Event Type")
        plt.ylabel("Total Duration (sec)")
        plt.tight_layout()
        total_duration_path = os.path.join(split_stats_dir, "total_duration_per_event.png")
        plt.savefig(total_duration_path)
        plt.close()
        print(f"Saved total duration bar chart for {split} at {total_duration_path}")

        # 2. Average Duration per Event Type
        avg_duration = df.groupby("event_type")["duration_sec"].mean().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        avg_duration.plot(kind="bar")
        plt.title(f"Average Duration per Event Type in {split.capitalize()} Split (sec)")
        plt.xlabel("Event Type")
        plt.ylabel("Average Duration (sec)")
        plt.tight_layout()
        avg_duration_path = os.path.join(split_stats_dir, "avg_duration_per_event.png")
        plt.savefig(avg_duration_path)
        plt.close()
        print(f"Saved average duration bar chart for {split} at {avg_duration_path}")

        # 3. Boxplot of Event Durations per Event Type
        plt.figure(figsize=(12, 8))
        df.boxplot(column="duration_sec", by="event_type", grid=False)
        plt.title(f"Distribution of Event Durations in {split.capitalize()} Split (sec)")
        plt.suptitle("")
        plt.xlabel("Event Type")
        plt.ylabel("Duration (sec)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        boxplot_path = os.path.join(split_stats_dir, "duration_boxplot_per_event.png")
        plt.savefig(boxplot_path)
        plt.close()
        print(f"Saved duration boxplot for {split} at {boxplot_path}")

        # 4. Scatter Plot: Start Time vs. Duration (all events)
        plt.figure(figsize=(10, 6))
        event_types = df["event_type"].unique()
        for event in event_types:
            subset = df[df["event_type"] == event]
            plt.scatter(subset["start_sec"], subset["duration_sec"], label=event, alpha=0.7)
        plt.title(f"Start Time vs Duration for Events in {split.capitalize()} Split")
        plt.xlabel("Start Time (sec)")
        plt.ylabel("Duration (sec)")
        plt.legend()
        plt.tight_layout()
        scatter_all_path = os.path.join(split_stats_dir, "start_time_vs_duration_all.png")
        plt.savefig(scatter_all_path)
        plt.close()
        print(f"Saved scatter plot for all events in {split} at {scatter_all_path}")

        # 5. Scatter Plot: Start Time vs. Duration for Each Video
        video_ids = df["video_id"].unique()
        for video in video_ids:
            subset_video = df[df["video_id"] == video]
            plt.figure(figsize=(10, 6))
            for event in subset_video["event_type"].unique():
                subset_event = subset_video[subset_video["event_type"] == event]
                plt.scatter(subset_event["start_sec"], subset_event["duration_sec"], label=event, alpha=0.7)
            plt.title(f"Start Time vs Duration for Video {video} in {split.capitalize()} Split")
            plt.xlabel("Start Time (sec)")
            plt.ylabel("Duration (sec)")
            plt.legend()
            plt.tight_layout()
            scatter_video_path = os.path.join(split_stats_dir, f"start_time_vs_duration_{video}.png")
            plt.savefig(scatter_video_path)
            plt.close()
            print(f"Saved scatter plot for video {video} in {split} at {scatter_video_path}")

if __name__ == "__main__":
    annotations_csv_dir = os.path.join("visualizations", "annotations_csv")
    output_stats_dir = os.path.join("visualizations", "statistics")
    generate_statistics(annotations_csv_dir, output_stats_dir)
