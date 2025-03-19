import cv2
import os
import argparse
import re
import csv

def parse_annotation_file(annotation_file):
    """
    Parses the annotation file.
    The file is assumed to have rows with columns:
    Event Label, Start, End, Duration (times in milliseconds)
    
    This function returns a dictionary mapping each event type to a list of intervals
    (start, end) in seconds.
    """
    annotations = {}
    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try splitting on one or more tabs.
            tokens = re.split(r'\t+', line)
            if len(tokens) < 4:
                # Fallback: split on whitespace.
                tokens = line.split()
                if len(tokens) < 4:
                    print(f"Skipping line (not enough tokens): {line}")
                    continue
                # Assume the last three tokens are start, end, duration.
                event = " ".join(tokens[:-3])
                start, end, _ = tokens[-3:]
            else:
                event = tokens[0].strip()
                start, end, _ = tokens[1:4]
            try:
                start = float(start) / 1000.0  # convert to seconds
                end = float(end) / 1000.0
            except ValueError:
                print(f"Skipping line (invalid numbers): {line}")
                continue
            if event not in annotations:
                annotations[event] = []
            annotations[event].append((start, end))
    return annotations

def classify_clip(clip_start, clip_end, annotations, event_types):
    """
    For a clip between clip_start and clip_end (in seconds), compute a classification vector.
    
    For each event type, this function sums the total duration (in seconds) during which the
    event is active within the clip (using all annotated intervals for that event). If the
    total overlap is at least half of the clip duration, the event is classified as present (1),
    otherwise as absent (0).
    """
    clip_duration = clip_end - clip_start
    classification = []
    for event in event_types:
        intervals = annotations.get(event, [])
        total_overlap = 0.0
        for (event_start, event_end) in intervals:
            # Compute overlap between [clip_start, clip_end] and [event_start, event_end]
            overlap = max(0, min(clip_end, event_end) - max(clip_start, event_start))
            total_overlap += overlap
        # Mark as 1 if the event occurs for at least half of the clip duration.
        classification.append(1 if total_overlap >= (clip_duration / 2.0) else 0)
    return classification

def extract_clip(video_path, clip_start, clip_end, output_path, fps, frame_size):
    """
    Extracts a clip from the video between clip_start and clip_end (in seconds)
    and saves it to output_path using OpenCV.
    """
    cap = cv2.VideoCapture(video_path)
    start_frame = int(clip_start * fps)
    end_frame = int(clip_end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Define the codec and create VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1
    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(description="Divide video into overlapping clips and classify events.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file (mp4).")
    parser.add_argument("--annotations", type=str, required=True, help="Path to the annotation file.")
    parser.add_argument("--clip_duration", type=float, default=10.0, help="Duration of each clip in seconds.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Fraction of overlap between consecutive clips (e.g., 0.5 for 50%% overlap).")
    parser.add_argument("--output_dir", type=str, default="clips", help="Directory to save the output clips and CSV file.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse the annotation file.
    annotations = parse_annotation_file(args.annotations)
    # Determine event types (here sorted alphabetically; you can change this order if needed).
    event_types = sorted(annotations.keys())
    print("Detected event types:", event_types)
    
    # Open the video and get its properties.
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    cap.release()
    
    # Calculate the step (in seconds) between clip starts.
    # For 50% overlap, each new clip starts after clip_duration/2 seconds.
    step = args.clip_duration * (1 - args.overlap)
    clip_data = []  # List of tuples: (clip_filename, clip_start, clip_end, classification_vector)
    clip_index = 0
    clip_start = 0.0
    while clip_start < video_duration:
        clip_end = clip_start + args.clip_duration
        # Ensure we do not exceed the video duration.
        if clip_end > video_duration:
            clip_end = video_duration
        clip_filename = os.path.join(args.output_dir, f"clip_{clip_index:04d}.mp4")
        
        # Extract the clip from the video.
        extract_clip(args.video, clip_start, clip_end, clip_filename, fps, frame_size)
        # Classify the clip.
        classification = classify_clip(clip_start, clip_end, annotations, event_types)
        clip_data.append((clip_filename, clip_start, clip_end, classification))
        print(f"Processed clip {clip_index}: {clip_filename}, time {clip_start:.2f}-{clip_end:.2f}, classification: {classification}")
        
        clip_index += 1
        clip_start += step  # move to next clip start
    
    # Save the clip classification data to a CSV file.
    csv_filename = os.path.join(args.output_dir, "clip_annotations.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["clip_filename", "clip_start", "clip_end"] + event_types
        writer.writerow(header)
        for (filename, start, end, classification) in clip_data:
            row = [filename, start, end] + classification
            writer.writerow(row)
    print(f"Saved clip annotations to {csv_filename}")

if __name__ == "__main__":
    main()
