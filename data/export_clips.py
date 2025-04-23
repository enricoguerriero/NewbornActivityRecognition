import os
import cv2
import numpy as np

def export_clips(video_folder, annotation_folder, output_folder, clip_length, overlapping, frame_per_second):
    
    video_files = os.listdir(video_folder)
    annotation_files = os.listdir(annotation_folder)
    print(f"Found {len(video_files)} video files and {len(annotation_files)} annotation files.")
    
    for i, _ in enumerate(video_files):
        
        print("-" * 20)
        print(f"Processing video {i + 1}/{len(video_files)}")
        print("-" * 20)
        
        video_file = os.path.join(video_folder, video_files[i])
        annotation_file = os.path.join(annotation_folder, annotation_files[i])
        print(f"Processing {video_file} and {annotation_file}")
        annotation = read_annotations(annotation_file)
        
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error opening video file {video_file}")
            continue
    
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video FPS: {fps}")
        print(f"Expected FPS: {frame_per_second}")
        frame_interval = int(fps / frame_per_second)
        print(f"Frame interval: {frame_interval}")
        
        frame_per_clip = int(clip_length * frame_per_second)
        overlapping_frames = int(overlapping * frame_per_clip)
        overlapping_clips = int(frame_per_clip / (frame_per_clip - overlapping_frames))
        print(f"Frame per clip: {frame_per_clip}")
        print(f"Overlapping frames: {overlapping_frames}")
        print(f"Overlapping clips: {overlapping_clips}")
        
        frame_index = 0
        frames_list = []
        clip_index = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_index % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(frame)
                
                if len(frames_list) == frame_per_clip:
                    label = label_clip(frame_index * fps * 1000, clip_length * 1000, annotation)
                    label_str = "_".join(map(str, label))
                    file_name = "video_" + str(i) + "_clip_" + str(clip_index) + label_str + ".npy"
                    np.save(os.path.join(output_folder, file_name), np.array(frames_list))
                    frames_list = frames_list[overlapping_frames:]
                    
                    clip_index += 1
            frame_index += 1
            
        cap.release()
        print(f"Finished processing {video_file}")
        print(f"Exported {clip_index} clips from {video_file}")
        print("-" * 20)
        
def read_annotations(file_path):
    """
    Reads an annotation .txt file and returns a list of tuples:
      (label: str, start: int, end: int, length: int)

    Assumes each line has a variable-length label followed by three integer fields.
    """
    annotations = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # Find first numeric token; tokens before that form the label
            idx = next((i for i, p in enumerate(parts) if p.isdigit()), None)
            if idx is None or idx + 2 >= len(parts):
                continue  # skip malformed lines
            label = ' '.join(parts[:idx])
            ann_start = int(parts[idx])
            ann_end = int(parts[idx + 1])
            ann_length = int(parts[idx + 2])
            annotations.append((label, ann_start, ann_end, ann_length))
    return annotations
        

def label_clip(clip_start, clip_length, annotations):
    """
    Label the clip based on the annotations.
    """
    # Initialize labels and overlap accumulators
    labels = [0, 0, 0, 0]
    overlap = [0, 0, 0, 0]

    clip_end = clip_start + clip_length

    # Mapping of annotation labels to indices
    category_map = {
        'Baby visible': 0,
        'CPAP': 1, 'PPV': 1,
        'Stimulation trunk': 2, 'Stimulation back/nates': 2,
        'Suction': 3
    }

    for ann in annotations:
        ann_label, ann_start, ann_end, _ = ann
        if ann_label not in category_map:
            continue
        # Compute overlap duration
        start_overlap = max(clip_start, ann_start)
        end_overlap = min(clip_end, ann_end)
        dur = end_overlap - start_overlap
        if dur > 0:
            idx = category_map[ann_label]
            overlap[idx] += dur

    # Determine labels based on >50% coverage
    threshold = clip_length / 2
    for i in range(len(labels)):
        if overlap[i] > threshold:
            labels[i] = 1

    return labels