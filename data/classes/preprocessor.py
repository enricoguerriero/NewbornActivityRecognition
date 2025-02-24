import os
import cv2
import torch
import numpy as np
from data.utils.utils import collect_event_categories
from tqdm import tqdm


class ClipPreprocessor:
    def __init__(self, video_folder, annotation_folder, output_folder,
                 clip_length=5, frames_per_second=2, overlap=0, target_size=(224, 224),
                 transform=None):
        """
        Args:
            video_folder (str): Folder with input videos.
            annotation_folder (str): Folder with annotation files.
            output_folder (str): Folder to save the generated clip .pt files.
            clip_length (float): Duration of each clip in seconds.
            frames_per_second (int): Number of frames to sample per second.
            overlap (float): Overlapping duration between consecutive clips (in seconds).
            target_size (tuple): (width, height) to which images are resized.
            transform (callable, optional): Optional transformation applied to each frame.
        """
        self.video_folder = video_folder
        self.annotation_folder = annotation_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.clip_length = clip_length
        self.frames_per_second = frames_per_second
        self.overlap = overlap
        self.target_size = target_size
        self.transform = transform
        self.event_categories = collect_event_categories(annotation_folder)

    def preprocess_all(self, logger=None):
        """
        Process all videos in the video_folder. Assumes the annotation file shares
        the same base name as the video (with a .txt extension).
        """
        video_files = sorted(os.listdir(self.video_folder))
        logger.debug(f"Found {len(video_files)} video files.")
        logger.debug(f"Event categories: {self.event_categories}")
        for video_file in tqdm(video_files):
            video_path = os.path.join(self.video_folder, video_file)
            base_name, _ = os.path.splitext(video_file)
            annotation_path = os.path.join(self.annotation_folder, base_name + ".txt")
            if not os.path.exists(annotation_path):
                logger.error(f"Annotation file for {video_file} not found, skipping.")
                continue
            self._preprocess_video(video_path, annotation_path, logger)

    def _preprocess_video(self, video_path, annotation_path, logger=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.error(f"Invalid FPS for {video_path}, using fallback of 30")
            fps = 30

        # Determine sampling parameters.
        frame_interval = int(round(fps / self.frames_per_second))
        clip_frame_count = int(self.clip_length * self.frames_per_second)
        overlap_frames = int(self.overlap * self.frames_per_second)
        if overlap_frames >= clip_frame_count:
            logger.error("Overlap must be less than clip length. Setting overlap to 0.")
            overlap_frames = 0
        hop_frames = clip_frame_count - overlap_frames

        clip_index = 0
        # current_clip stores tuples: (processed_frame, absolute_frame_index)
        current_clip = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                # Process the frame.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame_proc = self.transform(frame_rgb)
                else:
                    frame_resized = cv2.resize(frame_rgb, self.target_size)
                    frame_proc = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
                current_clip.append((frame_proc, frame_idx))
                if len(current_clip) == clip_frame_count:
                    # Get list of absolute frame indices for this clip.
                    frame_indices = [item[1] for item in current_clip]
                    clip_start_time = (frame_indices[0] / fps) * 1000  # in milliseconds
                    # Generate per-frame binary labels and then average to get clip-level label.
                    labels = self._generate_labels(annotation_path, frame_indices, fps, logger)
                    clip_labels = torch.mean(labels.float(), dim=0)
                    # Stack frames into a tensor of shape [num_frames, C, H, W].
                    frames_tensor = torch.stack([item[0] for item in current_clip])
                    clip_data = {
                        'frames': frames_tensor,
                        'labels': clip_labels,
                        'video_path': video_path,
                        'clip_index': clip_index,
                        'clip_start_time': clip_start_time,
                        'clip_length': self.clip_length,
                        'sampling_rate': self.frames_per_second
                    }
                    output_filename = os.path.splitext(os.path.basename(video_path))[0] + f"_clip_{clip_index}.pt"
                    output_filepath = os.path.join(self.output_folder, output_filename)
                    torch.save(clip_data, output_filepath)
                    logger.debug(f"Saved clip: {output_filepath}")
                    clip_index += 1
                    # Slide the window: keep the last 'overlap_frames' frames.
                    current_clip = current_clip[hop_frames:]
            frame_idx += 1
        cap.release()

    def _generate_labels(self, annotation_path, frame_indices, fps, logger = None):
        """
        Generate per-frame labels for a clip given the absolute frame indices.
        Returns a tensor of shape [num_frames, num_events] with binary values.
        """
        num_frames = len(frame_indices)
        num_categories = len(self.event_categories)
        labels = np.zeros((num_frames, num_categories), dtype=np.int64)
        # Compute the absolute time in milliseconds for each frame.
        frame_times = np.array([(idx / fps) * 1000 for idx in frame_indices])
        try:
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading annotation file {annotation_path}: {e}")
            return torch.tensor(labels, dtype=torch.int64)

        for line in lines:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 4:
                continue
            event_name = " ".join(tokens[:-3]).strip()
            try:
                event_start = float(tokens[-3])
                event_end = float(tokens[-2])
            except ValueError:
                continue
            if event_name in self.event_categories:
                event_idx = self.event_categories.index(event_name)
                indices = np.where((frame_times >= event_start) & (frame_times < event_end))[0]
                labels[indices, event_idx] = 1
        return torch.tensor(labels, dtype=torch.int64)