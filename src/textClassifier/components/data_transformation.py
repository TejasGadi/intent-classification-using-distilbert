## Data Transfomation operation libraries import

import os
from src.textClassifier.logging import logger
from transformers import AutoTokenizer 
from datasets import load_dataset
from src.textClassifier.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig):
        self.config = data_transformation_config

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        self.label2id = {
            "Intent_Lease_Abstraction": 0,
            "Intent_Comparison_LOI_Lease": 1,
            "Intent_Clause_Protect": 2,
            "Intent_Company_research": 3,
            "Intent_Transaction_Date_navigator": 4,
            "Intent_Amendment_Abstraction": 5,
            "Intent_Sales_Listings_Comparison": 6,
            "Intent_Lease_Listings_Comparison": 7,
        }

        self.id2label = {v: k for k, v in self.label2id.items()}

    # Preprocessing function with label mapping
    def preprocess_function(self,examples):
        # Tokenize the text field
        tokenized = self.tokenizer(examples['email_text'], padding="max_length", truncation=True)

        # Convert class labels to integers
        tokenized["labels"] = [self.label2id[label] for label in examples["intent"]]
        return tokenized


    def apply_transformation_to_dataset(self):
        # Load local CSV file
        raw_dataset = load_dataset(
            'csv',
            data_files={
                'train': self.config.local_train_csv_path,
                'validation': self.config.local_test_csv_path
            }
        )


        # Apply preprocessing
        encoded_dataset = raw_dataset.map(self.preprocess_function, batched=True)

        encoded_dataset.save_to_disk(os.path.join(self.config.root_dir, "intent-dataset"))