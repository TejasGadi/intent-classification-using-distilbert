from src.textClassifier.config.configuration import ConfigurationManager
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        
        # Load model and tokenizer from the configured local path
        self.model = DistilBertForSequenceClassification.from_pretrained(str(self.config.model_path))
        self.tokenizer = DistilBertTokenizer.from_pretrained(str(self.config.tokenizer_path))

        # Set device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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

    def predict(self, text: str):
        # Tokenize input text and move tensors to correct device
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        return {"text": text, "prediction": prediction, "class_label": self.id2label[prediction]}

