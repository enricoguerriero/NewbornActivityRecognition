import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from tqdm import tqdm
from torchvision import transforms
from data.utils import LeftCrop, pad_or_truncate

class VideoDataset(Dataset):
    def __init__(self, video_folder: str, annotation_folder: str, 
                 clip_length: float, overlapping: float, size: tuple, 
                 frames_per_second: int, tensors=False, event_categories: list[str] = [],
                 exploring: bool = False, processor = None, model_name = None, tensor_folder = None):
        self.video_folder = video_folder
        self.video_list = sorted(os.listdir(video_folder))
        self.annotation_folder = annotation_folder
        self.annotation_list = sorted(os.listdir(annotation_folder))
        self.tensor_folder = tensor_folder
        self.clip_length = clip_length
        self.overlapping = overlapping
        self.size = size
        self.frames_per_second = frames_per_second
        self.processor = processor
        self.video_tensors = None
        if tensors:
            # Check if the video tensors already exist and load them.
            self.get_video_tensors(model_name)
                
        self.event_categories = event_categories if event_categories else ["Baby visible", "Ventilation", "Stimulation", "Suction"]
        
        if exploring:
            # Additional exploration functionality can be added here.
            pass
        
        # Dummy cache system to store previously loaded clips.
        self.cache = {}
        
        self.index_mapping = self.index_mapping_creation()
        
    def __len__(self):
        # Total number of clips computed from all videos.
        return len(self.index_mapping)
    
    def _load_clip_data_from_cache(self, clip_name):
        """
        Internal helper to load clip data from a cache.
        """
        if clip_name in self.cache:
            return self.cache[clip_name]
        return None

    def __getitem__(self, idx):
        """
        Loads and returns video clip data for a given global index.
        """
        # Retrieve the corresponding video and clip indices from our mapping.
        video_idx, clip_idx = self.index_mapping[idx]
        video = self.video_list[video_idx]
        annotation = self.annotation_list[video_idx]  # Assuming sorting aligns videos with annotations.
        video_path = os.path.join(self.video_folder, video)
        annotation_path = os.path.join(self.annotation_folder, annotation)
        
        clip_name = f"{video}_{clip_idx}"
        
        # Check if clip data has been cached.
        clip_data = self._load_clip_data_from_cache(clip_name)
        if clip_data is not None:
            return clip_data
        
        # Load frames from precomputed tensors if available; otherwise, load from video file.
        if self.video_tensors is not None:
            frames = self.load_frames_from_tensors(video, clip_idx)
        # else:
        #     frames = self.load_frames(video_path, clip_idx)
        labels = self.load_labels(annotation_path, clip_idx)
        
        frames = pad_or_truncate(frames, length=int(self.clip_length * self.frames_per_second))
        clip_data = {
            'frames': frames,
            'labels': labels,
            'clip_name': clip_name
        }
        self.cache[clip_name] = clip_data
        return clip_data
    
    # def load_frames(self, video_path, clip_idx):
    #     """
    #     Load frames from a video clip file.
    #     """
    #     cap = cv2.VideoCapture(video_path)
    #     if not cap.isOpened():
    #         print(f"Error opening video file: {video_path}", flush=True)
    #         return None
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     start_frame = int(clip_idx * fps * (self.clip_length - self.overlapping))
    #     end_frame = int(start_frame + (self.clip_length * fps))
    #     frame_interval = int(round(fps / self.frames_per_second))
    #     frames = []
    #     frame_index = 0
        
    #     transform = transforms.Compose([
    #         transforms.ToPILImage(),
    #         LeftCrop(self.size),
    #         transforms.Resize(self.size),  
    #         transforms.ToTensor()
    #     ])
        
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         if frame_index % frame_interval == 0 and frame_index >= start_frame:
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             frame = transform(frame)
    #             frames.append(frame)
    #         frame_index += 1
    #         if frame_index >= end_frame:
    #             break
    #     cap.release()
    #     if frames:
    #         frames = torch.stack(frames)
    #     else:
    #         frames = None            
    #     return frames
    
    def load_frames_from_tensors(self, video, clip_idx):
        """
        Load frames from a precomputed video tensor.
        """
        video_tensor = self.video_tensors[video]
        start_frame = int(clip_idx * self.frames_per_second)
        end_frame = int(start_frame + (self.clip_length * self.frames_per_second))
        frames = video_tensor[start_frame:end_frame]
        return frames

    def load_labels(self, annotation_path, clip_idx):
        """
        Load labels for the clip from the annotation file.
        """
        clip_window_start = clip_idx * (self.clip_length - self.overlapping)
        clip_window_end = clip_window_start + self.clip_length
        
        clip_start_ms = clip_window_start * 1000
        clip_end_ms = clip_window_end * 1000
        
        labels = {cat: 0 for cat in self.event_categories}
        
        if len(self.event_categories) == 4:
            mapping = {
                "Baby visible": "Baby visible",
                "CPAP": "Ventilation",
                "PPV": "Ventilation",
                "Stimulation trunk": "Stimulation",
                "Stimulation back/nates": "Stimulation",
                "Suction": "Suction"
            }
        elif len(self.event_categories) == 7:
            mapping = {
                "Baby visible": "Baby visible",
                "CPAP": "CPAP",
                "PPV": "PPV",
                "Stimulation trunk": "Stimulation trunk",
                "Stimulation back/nates": "Stimulation back/nates",
                "Stimulation extremities": "Stimulation extremities",
                "Suction": "Suction",  
            }
        else:
            raise ValueError("Unsupported number of event categories. Supported: 4 or 7.")
        
        try:
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading annotation file {annotation_path}: {e}", flush=True)
            return torch.tensor(list(labels.values()), dtype=torch.float)

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
            mapped_label = mapping.get(event_name)
            if mapped_label is None:
                continue
            overlap = max(0, min(clip_end_ms, event_end) - max(clip_start_ms, event_start))
            proportion = overlap / (self.clip_length * 1000)
            labels[mapped_label] += proportion
        
        for cat in labels:
            labels[cat] = min(max(labels[cat], 0), 1)
        
        label_list = [labels[cat] for cat in self.event_categories]
        return torch.tensor(label_list, dtype=torch.float)
            
    def get_data_loader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0):
        """
        Returns a DataLoader instance for the current dataset.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def store_tensors(self):
        
        self.video_tensors = {}
        for video in tqdm(self.video_list, desc="Transforming videos into tensors"):
            video_path = os.path.join(self.video_folder, video)
            cap = cv2.VideoCapture(video_path)    
            frames = []
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(round(fps / self.frames_per_second))
            if frame_interval < 1:
                frame_interval = 1
            transform = transforms.Compose([
                transforms.ToPILImage(),
                LeftCrop(self.size),  
                transforms.Resize(self.size),  
                transforms.ToTensor()
            ])
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_index % frame_interval == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if self.processor is not None:
                        frame = self.processor(images=frame, return_tensors="pt")["pixel_values"][0].squeeze(0)
                    else:
                        frame = transform(frame)
                    frames.append(frame)
                frame_index += 1
            cap.release()
            if frames:
                video_tensor = torch.stack(frames)
                self.video_tensors[video] = video_tensor
            
    def index_mapping_creation(self):
        
        index_mapping = []
        
        for video_idx, video in enumerate(self.video_list):
            video_path = os.path.join(self.video_folder, video)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_length = num_frames / fps if fps > 0 else 0
            # Compute how many clips can be extracted
            num_clips = max(0, int((video_length - (self.clip_length - self.overlapping)) / (self.clip_length - self.overlapping)))
            for clip_idx in range(num_clips):
                index_mapping.append((video_idx, clip_idx))
            cap.release()
        return index_mapping

    def weight_computation(self):
        """
        Computes per-class weights for positive and negative examples by iterating
        over the dataset and aggregating the 'labels' from each sample.
        
        Returns:
            pos_weight (torch.Tensor): A tensor of shape (n_classes,) containing the 
                                    weight for positive examples per class.
            neg_weight (torch.Tensor): A tensor of shape (n_classes,) containing the 
                                    weight for negative examples per class.
        """
        eps = 1e-7  # small constant to avoid division by zero
        labels_list = []
        
        for i in range(len(self)):
            sample = self[i]
            labels_list.append(sample['labels'])
        
        labels_tensor = torch.stack(labels_list)
        
        pos_counts = torch.sum(labels_tensor, dim=0)
        
        total_samples = labels_tensor.shape[0]
        neg_counts = total_samples - pos_counts
        
        pos_weight = neg_counts.float() / (pos_counts.float() + eps)
        
        neg_weight = pos_counts.float() / (neg_counts.float() + eps)
        
        return pos_weight, neg_weight


    def get_video_tensors(self, model_name):
        
        # Check if the tensors already exist
        video_tensors_path = os.path.join(self.tensor_folder, model_name)
        if os.path.exists(video_tensors_path):
            self.video_tensors = torch.load(video_tensors_path)
            return
        # If not, create the directory and initialize the tensors
        os.makedirs(video_tensors_path, exist_ok=True)
        self.video_tensors = {}
        self.store_tensors()
        # Save the tensors to disk
        torch.save(self.video_tensors, video_tensors_path)