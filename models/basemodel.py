import torch.nn as nn
import logging
import os
from tqdm import tqdm
from torchvision.transforms import transforms
import cv2
import numpy as np
import torch

class BaseVideoModel(nn.Module):
    """
    An abstract base class for video models.
    Provides a common interface for training, inference, and last-layer modifications.
    """
    def __init__(self, model_name: str = "baseModel"):
        super(BaseVideoModel, self).__init__()
        self.model_name = model_name
        self.video_folder = "data/videos"
        self.annotation_folder = "data/annotations"
        self.output_folder = "data/processed/" + self.model_name
        self.transform = None
        self.image_processor = None
    
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward method.")

    def modify_last_layer(self, new_layer_config):
        """
        Modify the last layer(s) of the model.
        :param new_layer_config: A new layer or sequential block to replace the final layer(s).
        """
        raise NotImplementedError("Subclasses must implement modify_last_layer().")
    
    def preprocess_videos(self, set_name: str, clip_length: int = 3, frames_per_second: int = 5, overlap: float = 0.5, event_categories: list[str] = []):
        """
        Preprocess videos for a given dataset.
        :param set_name: Name of the dataset (e.g., 'train', 'validation', 'test').
        :param clip_length: Length of video clips in seconds.
        :param frames_per_second: Number of frames per second.
        :param overlap: Overlap between consecutive clips.
        :param event_categories: List of event categories to consider.
        """

        logger = logging.getLogger(f'{self.model_name}_preprocessing')
        logger.info(f"Preprocessing videos for {set_name} set.")
        
        logger.debug(f'Input parameters: {set_name}, {clip_length}, {frames_per_second}, {overlap}, {event_categories}')
        
        video_folder = os.path.join(self.video_folder, set_name)
        annotation_folder = os.path.join(self.annotation_folder, set_name)
        output_folder = os.path.join(self.output_folder, set_name)
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Output folder: {output_folder}")
        
        video_files = sorted(os.listdir(video_folder))
        logger.info(f"Found {len(video_files)} video files.")
        logger.info(f"Event categories: {event_categories}")
        for video_file in tqdm(video_files):
            video_path = os.path.join(video_folder, video_file)
            base_name, _ = os.path.splitext(video_file)
            annotation_path = os.path.join(annotation_folder, base_name + ".txt")
            if not os.path.exists(annotation_path):
                logger.error(f"Annotation file for {video_file} not found, skipping.")
                continue
            logger.debug(f"Preprocessing video: {video_file}")
            self._preprocess_video(video_path, annotation_path, output_folder, 
                                   clip_length=clip_length, frames_per_second=frames_per_second,
                                   overlap=overlap, event_categories=event_categories, logger=logger)
            
    def _preprocess_video(self, video_path, annotation_path, output_folder, 
                          clip_length=3, frames_per_second=5, overlap=0.5, event_categories=[],
                          logger=None):
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.error(f"Invalid FPS for {video_path}, using fallback of 30")
            fps = 30

        # Determine sampling parameters.
        frame_interval = int(round(fps / frames_per_second))
        clip_frame_count = int(clip_length * frames_per_second)
        overlap_frames = int(overlap * frames_per_second)
        if overlap_frames >= clip_frame_count:
            logger.error("Overlap must be less than clip length. Setting overlap to 0.")
            overlap_frames = 0
        hop_frames = clip_frame_count - overlap_frames

        clip_index = 0
        # current_clip stores tuples: (processed_frame, absolute_frame_index)
        current_clip = []
        frame_idx = 0

        logger.debug(f"Sampling parameters computed, starting frame extraction.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                # Process the frame.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_proc = self.process_frame(frame_rgb)
                current_clip.append((frame_proc, frame_idx))
                if len(current_clip) == clip_frame_count:
                    # Get list of absolute frame indices for this clip.
                    frame_indices = [item[1] for item in current_clip]
                    clip_start_time = (frame_indices[0] / fps) * 1000  # in milliseconds
                    # Generate per-frame binary labels and then average to get clip-level label.
                    labels = self._generate_labels(annotation_path, frame_indices, fps, event_categories, logger)
                    clip_labels = torch.mean(labels.float(), dim=0)
                    # Stack frames into a tensor of shape [num_frames, C, H, W].
                    frames_tensor = torch.stack([item[0] for item in current_clip])
                    clip_data = {
                        'frames': frames_tensor,
                        'labels': clip_labels,
                        'video_path': video_path,
                        'clip_index': clip_index,
                        'clip_start_time': clip_start_time,
                        'clip_length': clip_length,
                        'sampling_rate': frames_per_second
                    }
                    logger.debug(f"tensor dimensions: {frames_tensor.shape}")
                    output_filename = os.path.splitext(os.path.basename(video_path))[0] + f"_clip_{clip_index}.pt"
                    output_filepath = os.path.join(output_folder, output_filename)
                    torch.save(clip_data, output_filepath)
                    logger.debug(f"Saved clip: {output_filepath}")
                    clip_index += 1
                    # Slide the window: keep the last 'overlap_frames' frames.
                    current_clip = current_clip[hop_frames:]
            frame_idx += 1
        cap.release()

    
    def define_transformation(self, target_size: tuple[int, int]):
        """
        Define a transformation to apply to the input frames.
        :param target_size: Target size for the frames.
        :return: A transformation object.
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            # transforms.ToTensor()
        ])
        

    def process_frame(self, frame_rgb):
        """
        Process a single frame (RGB) before passing to the model.
        """
        frame_proc = self.transform(frame_rgb)
        if self.image_processor is not None:
            frame_proc = self.image_processor(frame_proc)
        else:
            frame_proc = torch.tensor(np.array(frame_proc)).permute(2, 0, 1).float() / 255.0
        return frame_proc
    
    
    def _generate_labels(self, annotation_path, frame_indices, fps, event_categories, logger = None):
        """
        Generate per-frame labels for a clip given the absolute frame indices.
        Returns a tensor of shape [num_frames, num_events] with binary values.
        """
        num_frames = len(frame_indices)
        num_categories = len(event_categories)
        labels = np.zeros((num_frames, num_categories), dtype=np.int64)
        # Compute the absolute time in milliseconds for each frame.
        frame_times = np.array([(idx / fps) * 1000 for idx in frame_indices])
        try:
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading annotation file {annotation_path}: {e}")
            return torch.tensor(labels, dtype=torch.int32)

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
            if event_name in event_categories:
                event_idx = event_categories.index(event_name)
                indices = np.where((frame_times >= event_start) & (frame_times < event_end))[0]
                labels[indices, event_idx] = 1
        return torch.tensor(labels, dtype=torch.int32)