import os
import torch
import torch.nn as nn
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration, TrainingArguments, Trainer
from data.vlm_dataset import ClipDataset

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