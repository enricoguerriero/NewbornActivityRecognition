import h5py
import numpy as np
import torch

def write_clips_to_hdf5(clips, hdf5_path):
    """
    Save a list of video clips to an HDF5 file.

    Args:
        clips (list): Each element is a dictionary with keys:
            - 'frames': NumPy array or tensor of shape (num_frames, channels, height, width)
            - 'labels': NumPy array or tensor of shape (num_labels,) (e.g., 7 booleans)
            - Optional metadata keys: 'video_path', 'clip_index', 'clip_start_time', 'clip_length', 'sampling_rate'
        hdf5_path (str): Path to the output HDF5 file.
    """
    with h5py.File(hdf5_path, 'w') as hf:
        for i, clip in enumerate(clips):
            grp = hf.create_group(f'clip_{i}')
            # Convert tensors to numpy if necessary
            frames = clip['frames']
            if torch.is_tensor(frames):
                frames = frames.numpy()
            labels = clip['labels']
            if torch.is_tensor(labels):
                labels = labels.numpy()
            # Save the main data with compression
            grp.create_dataset('frames', data=frames, compression="gzip")
            grp.create_dataset('labels', data=labels)
            # Save optional metadata as attributes
            for key in ['video_path', 'clip_index', 'clip_start_time', 'clip_length', 'sampling_rate']:
                if key in clip:
                    grp.attrs[key] = clip[key]
    print(f"Clips saved to {hdf5_path}")


import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class HDF5VideoDataset(Dataset):
    def __init__(self, hdf5_path, transform=None):
        """
        Args:
            hdf5_path (str): Path to the HDF5 file containing the clips.
            transform: Optional transform to be applied on the frames.
        """
        self.hdf5_path = hdf5_path
        self.transform = transform
        # List available clip groups by opening the file once
        with h5py.File(self.hdf5_path, 'r') as hf:
            self.clip_keys = list(hf.keys())

    def __len__(self):
        return len(self.clip_keys)

    def __getitem__(self, idx):
        # Open the file for each access (ensuring compatibility with multi-worker DataLoaders)
        with h5py.File(self.hdf5_path, 'r') as hf:
            grp = hf[self.clip_keys[idx]]
            frames = np.array(grp['frames'])
            labels = np.array(grp['labels'])
            # Optional: load additional metadata from attributes if needed
            metadata = {key: grp.attrs[key] for key in grp.attrs}
        
        # Convert to torch tensors
        frames = torch.from_numpy(frames)
        labels = torch.from_numpy(labels)

        # Apply any transformation
        if self.transform:
            frames = self.transform(frames)

        sample = {
            'frames': frames,
            'labels': labels,
            'metadata': metadata  # includes info like video_path, clip_index, etc.
        }
        return sample

    def get_data_loader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0):
        """
        Convenience function to get a DataLoader for the dataset.
        """
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
