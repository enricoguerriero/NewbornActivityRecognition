import torch
import torch.nn as nn
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor, TrainingArguments, Trainer
from models.basemodel import BaseVideoModel

class VideoLlavaClassifier(BaseVideoModel):
    def __init__(self, checkpoint_path: str = None, base_model_id: str = "LanguageBind/Video-LLaVA-7B-hf", device=None, num_classes=4):
        super().__init__()
        self.name = "llava_video_classifier"
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the processor and model
        self.processor = VideoLlavaProcessor.from_pretrained(base_model_id)
        if checkpoint_path:
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16
            ).to(self.device)

        # Assume the embedding size comes from the base language model inside
        hidden_size = self.model.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None):
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        pooled = outputs.last_hidden_state[:, 0]  # Use CLS or first token
        logits = self.classifier(pooled)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier[-1].out_features), labels.view(-1))
        else:
            loss = None
        
        return {"loss": loss, "logits": logits}



    def train_model(self, 
                    train_dataset,
                    eval_dataset,
                    data_collator,
                    output_dir: str = "models/saved/llavavideo_classifier",
                    per_device_train_batch_size=2,
                    per_device_eval_batch_size=2,
                    gradient_accumulation_steps=2,
                    num_train_epochs=5,
                    learning_rate=2e-5,
                    logging_dir="logs",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    fp16=True,
                    report_to="none"
                    ):
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_dir=logging_dir,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            fp16=fp16,
            report_to=report_to
        )
        
        trainer = Trainer(
            model=self,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        trainer.train()
        trainer.save_model(output_dir)