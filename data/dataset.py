import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class PreprocessedClipDataset(Dataset):
    def __init__(self, preprocessed_folder):
        """
        Args:
            preprocessed_folder (str): Folder where .pt clip files are stored.
        """
        self.clip_files = sorted([
            os.path.join(preprocessed_folder, f)
            for f in os.listdir(preprocessed_folder)
            if f.endswith('.pt')
        ])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.clip_files)

    def __getitem__(self, idx):
        clip_data = torch.load(self.clip_files[idx])
        frames = clip_data['frames']
        labels = clip_data['labels']
        return frames.to(self.device), labels.to(self.device)


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
            
            
    def get_data_loader(self, batch_size, num_workers):
        return DataLoader(
            self, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        ).squeeze()
