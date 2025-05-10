import os
## Model Trainer libraries import
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
from datasets import load_from_disk
from src.textClassifier.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.config = model_trainer_config

    # Training logic
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = DistilBertTokenizer.from_pretrained(self.config.model_ckpt)

        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=8)

        # loading the data
        encoded_dataset = load_from_disk(self.config.data_path)

        training_args = TrainingArguments(
            output_dir="./results",           # Directory for saving results
            eval_strategy="epoch",     # Evaluate at the end of each epoch
            learning_rate=5e-5,              # Initial learning rate
            per_device_train_batch_size=16,  # Batch size per GPU
            num_train_epochs=3,              # Number of epochs
            weight_decay=0.01,               # Regularization
            logging_dir="./logs",            # Directory for logs
            logging_steps=10                 # Log every 10 steps
        )

        trainer = Trainer(
            model=model,                          # The DistilBERT model
            args=training_args,                   # Training arguments
            train_dataset=encoded_dataset['train'],  # Training data
            eval_dataset=encoded_dataset['validation']  # Validation data
        )

        # Start training
        trainer.train()

        ## Save model
        model.save_pretrained("./finetuned_model")
        ## Save tokenizer
        tokenizer.save_pretrained("./finetuned_model")