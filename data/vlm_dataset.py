import torch
from data.clip_dataset import VideoDataset  # or wherever VideoDataset lives

class ClipDataset(VideoDataset):
    def __init__(
        self,
        video_folder: str,
        annotation_folder: str,
        clip_length: float,
        overlapping: float,
        size: tuple,
        frames_per_second: int,
        processor, 
        prompt_processor, # your multimodal processor
        prompt: str,            # text prompt you want to prepend to every clip
        tensors: bool = False,
        event_categories: list[str] = None,
        exploring: bool = False,
        model_name: str = None,
        tensor_folder: str = None,
        set_name: str = None,
    ):
        # Initialize everything in VideoDataset
        super().__init__(
            video_folder=video_folder,
            annotation_folder=annotation_folder,
            clip_length=clip_length,
            overlapping=overlapping,
            size=size,
            frames_per_second=frames_per_second,
            tensors=tensors,
            event_categories=event_categories or [],
            exploring=exploring,
            processor=processor,
            model_name=model_name,
            tensor_folder=tensor_folder,
            set_name=set_name,
        )
        self.prompt = prompt
        self.prompt_processor = prompt_processor  # ensure it's on self

    def __len__(self):
        # Inherited behavior is correct; you could also just omit this method.
        return super().__len__()

    def __getitem__(self, idx):
        video_data = super().__getitem__(idx)
        frames = video_data['frames']         # Tensor [T, C, H, W]
        labels = video_data['labels']         # Tensor [n_events]

        inputs = self.prompt_processor(
            text=[self.prompt],
            videos=[frames],
            return_tensors="pt",
            padding=True,
        )

        return {
            "input_ids":           inputs["input_ids"].squeeze(0),
            "attention_mask":      inputs["attention_mask"].squeeze(0),
            "pixel_values":        inputs["pixel_values_videos"].squeeze(0),
            "labels":              labels.long(),
        }
