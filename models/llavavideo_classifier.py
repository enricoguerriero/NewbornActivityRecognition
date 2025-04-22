import torch
import torch.nn as nn
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor, TrainingArguments, Trainer
from models.basemodel import BaseVideoModel
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

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

        hidden_size = self.model.get_input_embeddings().embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        
        self.image_processor = None
        self.pos_weights = None

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None):
        outputs = self.model(
            pixel_values_videos=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

        last_layer = outputs.hidden_states[-1]  # Use CLS or first token
        pooled = last_layer[:, 0, :]  # CLS token representation
        logits = self.classifier(pooled.float())
        
        
        if labels is not None:
            if self.pos_weights is not None:
                pos_weights = torch.tensor(self.pos_weights).to(self.device)
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            else:
                loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
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
            eval_strategy=evaluation_strategy,
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
        return trainer
        
    def prompt_definition(self, question: str, system_message: str = "You are a helpful assistant.", video: list = None):
        """
        Build the prompt text for a given question.
        Here, we follow the recommended prompt format for Video LLaVA.
        """
        prompt = f"USER: <video>\n{system_message}\n{question}\nASSISTANT:"
        
        return prompt
    
    def set_weights(self, weights):
        """
        Set the model weights from a given path.
        """
        self.pos_weights = weights
        
        
    def test_model(
        self,
        test_dataset,
        data_collator,
        batch_size: int = 2,
        threshold: float = 0.5,
    ):
        """
        Evaluate model on test_dataset and compute:
          - average loss
          - overall accuracy
          - per-label accuracy
          - per-label precision, recall, F1
        Returns a dict with all of the above.
        """
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=False
        )

        self.model.eval()
        losses = []

        all_labels_list = []
        all_preds_list = []

        with torch.no_grad():
            for batch in test_loader:
                pixel_values   = batch.get("pixel_values", None)
                input_ids      = batch.get("input_ids", None)
                attention_mask = batch.get("attention_mask", None)
                labels         = batch["labels"].to(self.device)

                # Move inputs to device
                if pixel_values is not None:
                    pixel_values = pixel_values.to(self.device)
                if input_ids is not None:
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)

                # Forward
                out = self.forward(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss   = out["loss"]
                logits = out["logits"]

                losses.append(loss.item())

                # Predictions
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).long()

                all_labels_list.append(labels.cpu().numpy())
                all_preds_list.append(preds.cpu().numpy())

        # Stack batches → shape (N_examples, num_classes)
        y_true = np.vstack(all_labels_list)
        y_pred = np.vstack(all_preds_list)

        # Overall accuracy (element‑wise)
        overall_acc = (y_pred == y_true).mean()

        # Per-class accuracy
        per_label_acc = ((y_pred == y_true).sum(axis=0) / y_true.shape[0]).tolist()

        # Precision, recall, F1 per label
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        metrics = {
            "loss": float(np.mean(losses)),
            "overall_accuracy": float(overall_acc),
            "per_label_accuracy": per_label_acc,
            "precision_per_label": precision.tolist(),
            "recall_per_label": recall.tolist(),
            "f1_per_label": f1.tolist(),
        }

        # Print summary
        print(f"[Test] Loss: {metrics['loss']:.4f}  Overall Acc: {metrics['overall_accuracy']:.4f}", flush = True)
        for idx, acc in enumerate(per_label_acc):
            print(f"  Class {idx} — Acc: {acc:.4f},  Prec: {precision[idx]:.4f},  Rec: {recall[idx]:.4f},  F1: {f1[idx]:.4f}", flush = True)

        return metrics
