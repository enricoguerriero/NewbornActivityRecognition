from transformers import VideoLlavaForConditionalGeneration
import torch.nn as nn

class VideoLlavaForClassification(VideoLlavaForConditionalGeneration):
    def __init__(self, config, num_classes):
        super().__init__(config)
        hidden_size = self.model.model.embed_tokens.embedding_dim  # or use `config.hidden_size`
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **kwargs
        )

        # Use pooled output / first token / CLS token
        pooled = outputs.last_hidden_state[:, 0]  # (batch_size, hidden_size)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits
        }
