from data.clip_dataset import VideoDataset
from torch.utils.data import Dataset
import torch


class ClipDataset(Dataset):

    def __init__(self, video_dataset: VideoDataset, prompt: str = "What is happening in this video clip?", processor=None):
        super().__init__()
        
        self.video_dataset = video_dataset
        self.processor = processor
        self.prompt = prompt
        
    def __len__(self):
        return len(self.video_dataset)
    
    def __getitem__(self, idx):
        video_data = self.video_dataset[idx]
        
        frames = video_data['frames']
        labels = video_data['labels']
       
        inputs = self.processor(text=self.prompt, images=frames, return_tensors="pt", padding=True)
        print(inputs.keys(), flush = True)
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long)
        }