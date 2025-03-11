import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from tqdm import tqdm

class VideoDataset(Dataset):
    def __init__(self, video_clips_folder: str, transform=None):
        """
        :param video_clips_folder: Folder of processed data. Each clip file should be a 
                            serialized dictionary with keys:
                              - 'frames': tensor of shape (num_frames, channels, height, width)
                              - 'labels': labels for each clip, 1D tensor with shape (num_classes)
                              - 'video_path': path of the video (optional)
                              - 'clip_index': index of the clip (optional)
                              - 'clip_start_time': start time of the clip (optional)
                              - 'clip_length': length of the clip (optional)
                              - 'sampling_rate': frames per second (optional)
        :param transform: Optional transform or augmentation to apply on the frames.
        """
        
        video_clips = sorted(os.listdir(video_clips_folder))
        video_clips = [os.path.join(video_clips_folder, clip) for clip in video_clips]
        self.video_clips = video_clips
        self.transform = transform
        
        # dummy cache system ! to check if it makes sense
        self.cache = {}

    def __len__(self):
        return len(self.video_clips)

    # still to decide if using this function (memory issues)
    def _load_clip_data(self, clip_path):
        """
        Internal helper to load clip data from a file and cache it.
        Assumes the file is saved via torch.save() as a dictionary.
        """
        if clip_path in self.cache:
            return self.cache[clip_path]
        else:
            clip_data = torch.load(clip_path)
            self.cache[clip_path] = clip_data
            return clip_data

    def __getitem__(self, idx):
        """
        Loads and returns a video clip sample along with its metadata.
        """
        clip_path = self.video_clips[idx]
        # if using the cache:
        # clip_data = self._load_clip_data(clip_path)
        clip_data = torch.load(clip_path)
        frames = clip_data['frames']
        labels = clip_data['labels']
        sample = {
            'frames': frames,
            'labels': labels,
            'video_path': clip_data.get('video_path', clip_path),
            'clip_index': clip_data.get('clip_index', idx),
            'clip_start_time': clip_data.get('clip_start_time'),
            'clip_length': clip_data.get('clip_length'),
            'sampling_rate': clip_data.get('sampling_rate')
        }
        return sample

    def load_frames(self, clip_path):
        """
        Load frames from a video clip file.
        """
        clip_data = torch.load(clip_path)
        frames = clip_data['frames']
        return frames

    def load_labels(self, clip_path):
        """
        Load labels from a video clip file.
        """
        clip_data = torch.load(clip_path)
        labels = clip_data['labels']
        return labels
    
    def get_data_loader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0):
        """
        Returns a DataLoader instance for the current dataset.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    def export_all_clips_to_mp4(self, export_folder, export_fps=None, codec='mp4v', label_list=[], logger=None):
        """
        Exports all .pt clip files as MP4 videos.
        
        Args:
            export_folder (str): Folder where MP4 files will be saved.
            export_fps (int, optional): Frames per second for the output video.
                                        If None, uses the stored sampling_rate.
            codec (str): FourCC code for video codec (default 'mp4v').
        """
        os.makedirs(export_folder, exist_ok=True)
        for pt_file in tqdm(self.clip_files):
            clip_data = torch.load(pt_file, weights_only=True)
            frames = clip_data['frames']  # Tensor of shape [num_frames, C, H, W]
            # Convert frames to numpy array in HxWxC format.
            frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()
            num_frames, H, W, C = frames_np.shape
            # Use stored sampling_rate if export_fps is not provided.
            if export_fps is None:
                export_fps = clip_data.get('sampling_rate', 2)
            # get the label of this clip
            label = clip_data['labels'] # Tensor of shape [num_labels]
            # Define the output filename.
            base_name = os.path.splitext(os.path.basename(pt_file))[0]
            for j, event in enumerate(label_list):
                if label[j] > 0.5:
                    base_name += f"_{event}"
            output_path = os.path.join(export_folder, base_name + ".mp4")
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, export_fps, (W, H))
            for frame in frames_np:
                # Convert frame from float [0,1] to uint8 [0,255].
                frame_uint8 = (frame * 255).astype(np.uint8)
                # Convert from RGB to BGR for OpenCV.
                frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            logger.debug(f"Exported {output_path}")