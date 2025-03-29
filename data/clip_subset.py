from torch.utils.data import Dataset, DataLoader

class VideoSubsetDataset(Dataset):
    def __init__(self, full_dataset, video_idx, clip_indexes):
        self.full_dataset = full_dataset
        self.video_idx = video_idx
        self.clip_indexes = clip_indexes
        
    def __len__(self):
        return len(self.clip_indexes)
    
    def __getitem__(self, idx):
        # idx here is an index into the clip_indexes list
        clip_idx = self.clip_indexes[idx]
        return self.full_dataset[(self.video_idx, clip_idx)]