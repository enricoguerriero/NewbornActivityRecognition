from torch.utils.data import Dataset
from data.clip_dataset import VideoDataset
from models.basemodel import BaseVideoModel


class TokenDataset(Dataset):
    
    def __init__(self, clip_folder, model):
        
