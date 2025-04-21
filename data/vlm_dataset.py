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
       
        inputs = self.processor(text=self.prompt, videos=frames, return_tensors="pt", padding=True)
        print(f"Inputs: {inputs}", flush=True)
        print(f"Input keys: {inputs.keys()}", flush=True)
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values_videos": inputs["pixel_values_videos"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        
    
    def collate_fn(self, batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "pixel_values_videos": torch.stack([item["pixel_values_videos"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }