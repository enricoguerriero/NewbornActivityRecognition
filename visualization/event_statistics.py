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
    
    # Create output directory for statistics for the current split
    split_stats_dir = os.path.join(STATISTICS_OUTPUT_DIR, split)
    os.makedirs(split_stats_dir, exist_ok=True)
    
    # -- Bar Chart: Count of events per type --
    event_counts = df["event_type"].value_counts()
    plt.figure(figsize=(10, 6))
    event_counts.plot(kind="bar")
    plt.title(f"Event Counts in {split.capitalize()} Split")
    plt.xlabel("Event Type")
    plt.ylabel("Count")
    plt.tight_layout()
    bar_chart_path = os.path.join(split_stats_dir, "event_counts.png")
    plt.savefig(bar_chart_path)
    plt.close()
    print(f"Saved event counts bar chart for {split} at {bar_chart_path}")
    
    # -- Histogram: Distribution of event durations (in frames) --
    plt.figure(figsize=(10, 6))
    plt.hist(df["duration_frames"], bins=30)
    plt.title(f"Distribution of Event Durations in {split.capitalize()} Split (frames)")
    plt.xlabel("Duration (frames)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    histogram_path = os.path.join(split_stats_dir, "event_duration_histogram.png")
    plt.savefig(histogram_path)
    plt.close()
    print(f"Saved event duration histogram for {split} at {histogram_path}")
