from fastapi import FastAPI
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

app = FastAPI()

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./fine_tuned_model")
tokenizer = DistilBertTokenizer.from_pretrained("./fine_tuned_model")

@app.post("/predict/")
def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(axis=1).item()
    return {"text": text, "prediction": predictions}