import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from tqdm import tqdm
from torchvision import transforms

class VideoDataset(Dataset):
    def __init__(self, video_folder: str, annotation_folder: str, 
                 clip_length: float, overlapping: float, size: tuple, 
                 frames_per_second: int, tensors = False, event_categories: list[str] = []):

        self.video_folder = video_folder
        self.video_list = sorted(os.listdir(video_folder))
        self.annotation_folder = annotation_folder
        self.annotation_list = sorted(os.listdir(annotation_folder))
        self.clip_length = clip_length
        self.overlapping = overlapping
        self.size = size
        self.frames_per_second = frames_per_second
        if tensors:
            # transform the entire videos now and save them as tensors
            self.video_tensors = {}
            for video in tqdm(self.video_list, desc = "Transforming videos into tensors"):
                video_path = os.path.join(video_folder, video)
                cap = cv2.VideoCapture(video_path)    
                frames = []
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_interval = int(round(fps / self.frames_per_second))
                print(f"FPS: {fps}, fps desired: {self.frames_per_second}, Frame Interval: {frame_interval}")
                if frame_interval < 1:
                    frame_interval = 1
                
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(self.size),  
                    transforms.CenterCrop(self.size),  
                    transforms.ToTensor()
                ])
                
                frame_index = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_index % frame_interval == 0:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = transform(frame)
                        frames.append(frame)
                    
                    frame_index += 1
                
                if frames:
                    video_tensor = torch.stack(frames)
                    self.video_tensors[video] = video_tensor
                cap.release()
        else:
            self.video_tensors = None
        
        self.event_categories = event_categories if event_categories else ["Baby visible", "Ventilation", "Stimulation", "Suction"]
        
        # dummy cache system ! to check if it makes sense
        self.cache = {}

    def __len__(self):
        num_clips = 0
        # compute the number of clips for each video in the folder
        for video in self.video_list:
            video_path = os.path.join(self.video_folder, video)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_length = num_frames / fps
            num_clips += video_length - (self.clip_length - self.overlapping)
            cap.release()
        return num_clips

    # still to decide if using this function (memory issues)
    def _load_clip_data_from_cache(self, clip_name):
        """
        Internal helper to load clip data from a file and cache it.
        Assumes the file is saved via torch.save() as a dictionary.
        """
        if clip_name in self.cache:
            return self.cache[clip_name]
        else:
            return None

    def __getitem__(self, idx):
        """
        Loads and returns video clip data
        """
        video_idx, clip_idx = idx
        video = self.video_list[video_idx]
        annotation = self.annotation_list[video_idx]
        video_path = os.path.join(self.video_folder, video)
        annotation_path = os.path.join(self.annotation_folder, annotation)
        
        clip_name = f"{video}_{clip_idx}"
        
        # check if it is in cache
        clip_data = self._load_clip_data_from_cache(clip_name)
        if clip_data is not None:
            return clip_data
        
        if self.video_tensors is not None:
            frames = self.load_frames_from_tensors(video, clip_idx)
        else:
            frames = self.load_frames(video_path, clip_idx)
        labels = self.load_labels(annotation_path, clip_idx)
        clip_data = {
            'frames': frames,
            'labels': labels,
            'clip_name': clip_name} 
        self.cache[clip_name] = clip_data       
        
        return clip_data

    def load_frames(self, video_path, clip_idx):
        """
        Load frames from a video clip file.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}", flush=True)
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(clip_idx * fps * (self.clip_length - self.overlapping))
        end_frame = int(start_frame + (self.clip_length * fps))
        frame_interval = int(round(fps / self.frames_per_second))
        frames = []
        frame_index = 0
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size),  
            transforms.CenterCrop(self.size),  
            transforms.ToTensor()
        ])
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % frame_interval == 0 and frame_index >= start_frame:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = transform(frame)
                frames.append(frame)
            frame_index += 1
            if frame_index >= end_frame:
                break
        cap.release()
        if frames:
            frames = torch.stack(frames)
        else:
            frames = None            
        return frames
    
    def load_frames_from_tensors(self, video, clip_idx):
        """
        Load frames from a video clip file.
        """
        video_tensor = self.video_tensors[video]
        start_frame = int(clip_idx * self.frames_per_second)
        end_frame = int(start_frame + (self.clip_length * self.frames_per_second))
        frames = video_tensor[start_frame:end_frame]
        return frames

    def load_labels(self, annotation_path, clip_idx):
        """
        Load labels from a video clip file.
        """
        clip_window_start = clip_idx * (self.clip_length - self.overlapping)
        clip_window_end = clip_window_start + self.clip_length
        
        clip_start_ms = clip_window_start * 1000
        clip_end_ms = clip_window_end * 1000
        
        labels = {cat: 0 for cat in self.event_categories}
        
        # assuming event categories can only be 4 or 7
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
            mapped_label = mapping.get(event_name)
            
            overlap = max(0, min(clip_end_ms, event_end) - max(clip_start_ms, event_start))
            proportion = overlap / (self.clip_length * 1000)
            
            labels[mapped_label] += proportion
    
        for cat in labels:
            labels[cat] = min(max(labels[cat], 0), 1) # to be sure it is between 0 and 1
        
        label_list = [labels[cat] for cat in self.event_categories]
        return torch.tensor(label_list, dtype=torch.float)
            
    
    def get_data_loader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0):
        """
        Returns a DataLoader instance for the current dataset.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
