## Model Evaluation libraries import
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from tqdm import tqdm
import torch
from sklearn.metrics import classification_report

from src.textClassifier.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig):
        self.config = model_evaluation_config

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSequenceClassification.from_pretrained(self.config.model_path)

        # Load the dataset
        encoded_dataset = load_from_disk(self.config.data_path)

        validation_data = encoded_dataset['validation']
        texts = validation_data['email_text']

         # Tokenize the input text
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
    

         # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_labels = np.argmax(logits.cpu().numpy(), axis=1)
        
        true_labels = encoded_dataset['validation']['labels']

        # Generate classification report
        class_report = classification_report(true_labels, predicted_labels, output_dict=True)
        
        # Convert classification report to DataFrame for saving
        report_df = pd.DataFrame(class_report).transpose()

        # Save the report to CSV
        report_df.to_csv(self.config.metric_file_path, index=True)

        print(f"Evaluation Metrics saved to {self.config.metric_file_path}")