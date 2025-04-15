import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration, TrainingArguments, Trainer
from torchvision import transforms
import pandas as pd

# 👇 Modify this
NUM_CLASSES = 4
BASE_MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"
FIXED_PROMPT = "What is happening in this video clip?"
FRAME_LIMIT = 8

class VideoLlavaClassifier(VideoLlavaForConditionalGeneration):
    def __init__(self, config, num_classes):
        super().__init__(config)
        hidden_size = self.model.model.embed_tokens.embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        pooled = outputs.last_hidden_state[:, 0]  # CLS token or first token
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}

class ClipDataset(Dataset):
    def __init__(self, csv_file, processor, prompt=FIXED_PROMPT, frame_limit=8):
        self.data = pd.read_csv(csv_file)
        self.processor = processor
        self.prompt = prompt
        self.frame_limit = frame_limit
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def load_frames(self, video_path):
        frames = sorted(os.listdir(video_path))[:self.frame_limit]
        images = [self.transform(Image.open(os.path.join(video_path, f)).convert("RGB")) for f in frames]
        return images

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        frames = self.load_frames(row['video_path'])
        inputs = self.processor(text=self.prompt, images=frames, return_tensors="pt", padding=True)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(row['label'], dtype=torch.long)
        }

def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }

def main():
    processor = VideoLlavaProcessor.from_pretrained(BASE_MODEL_ID)
    model = VideoLlavaClassifier.from_pretrained(BASE_MODEL_ID, num_classes=NUM_CLASSES, torch_dtype=torch.float16).cuda()

    train_dataset = ClipDataset(csv_file="train.csv", processor=processor)
    val_dataset = ClipDataset(csv_file="val.csv", processor=processor)

    training_args = TrainingArguments(
        output_dir="./llava_video_classification",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        learning_rate=2e-5,
        logging_dir="./logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn
    )

    trainer.train()

if __name__ == "__main__":
    main()