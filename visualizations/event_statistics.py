import os
import pandas as pd
import matplotlib.pyplot as plt

# Directories
ANNOTATIONS_CSV_DIR = "visualizations/annotations_csv"
STATISTICS_OUTPUT_DIR = "visualizations/statistics"
os.makedirs(STATISTICS_OUTPUT_DIR, exist_ok=True)

splits = ["train", "validation", "test"]

for split in splits:
    csv_path = os.path.join(ANNOTATIONS_CSV_DIR, f"{split}_annotations.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file for split {split} not found: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    
    # Convert time values from milliseconds to seconds for easier interpretation.
    # Note: the CSV columns 'start_frame' and 'duration_frames' actually contain time in ms.
    df["start_sec"] = df["start_frame"] / 1000.0
    df["duration_sec"] = df["duration_frames"] / 1000.0
    
    # Create output directory for statistics for the current split
    split_stats_dir = os.path.join(STATISTICS_OUTPUT_DIR, split)
    os.makedirs(split_stats_dir, exist_ok=True)
    
    # --------------------------------------------------------
    # Plot 1: Bar Chart of Total Duration per Event Type (in seconds)
    # --------------------------------------------------------
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
    
    # --------------------------------------------------------
    # Plot 2: Bar Chart of Average Duration per Event Type (in seconds)
    # --------------------------------------------------------
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
    
    # --------------------------------------------------------
    # Plot 3: Boxplot of Event Durations per Event Type (in seconds)
    # --------------------------------------------------------
    plt.figure(figsize=(12, 8))
    df.boxplot(column="duration_sec", by="event_type", grid=False)
    plt.title(f"Distribution of Event Durations in {split.capitalize()} Split (sec)")
    plt.suptitle("")  # Remove the default suptitle added by pandas
    plt.xlabel("Event Type")
    plt.ylabel("Duration (sec)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    boxplot_path = os.path.join(split_stats_dir, "duration_boxplot_per_event.png")
    plt.savefig(boxplot_path)
    plt.close()
    print(f"Saved duration boxplot for {split} at {boxplot_path}")
    
    # --------------------------------------------------------
    # Plot 4: Scatter Plot of Start Time vs Duration (in seconds)
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    # Plot each event type separately for better clarity.
    event_types = df["event_type"].unique()
    for event in event_types:
        subset = df[df["event_type"] == event]
        plt.scatter(subset["start_sec"], subset["duration_sec"], label=event, alpha=0.7)
    plt.title(f"Start Time vs Duration for Events in {split.capitalize()} Split")
    plt.xlabel("Start Time (sec)")
    plt.ylabel("Duration (sec)")
    plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(split_stats_dir, "start_time_vs_duration.png")
    plt.savefig(scatter_path)
    plt.close()
    print(f"Saved start time vs duration scatter plot for {split} at {scatter_path}")
