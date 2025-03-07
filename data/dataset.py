import torch
from torch.utils.data import Dataset, DataLoader
import os

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