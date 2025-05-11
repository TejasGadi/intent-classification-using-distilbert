# 🧠 Intent Classification by finetuning DistilBert 🚀

This project is an end-to-end **text classification system** built using **FastAPI**, **Transformers**, **PyTorch**, and **Docker**, structured with clean, modular, and scalable MLOps-inspired components.

## 📌 Use Case
Classify user queries or emails into specific intents or categories  for real estate and legal document analysis use cases using a transformer-based model (e.g., `distilbert-base-uncased`).

## 📁 Project Structure
```
textClassifier/
│
├── src/textClassifier/ # Main Python package
│   ├── components/     # Data processing, model training/evaluation components
│   ├── config/         # Configuration management
│   ├── constants/      # Constants used across modules
│   ├── entity/         # Data classes (input/output/config schemas)
│   ├── pipeline/       # Stage-wise pipeline scripts
│   ├── utils/          # Utility functions
│   └── logging/        # Logging setup
│
├── config/             # YAML configuration files
│   └── config.yaml
├── params.yaml         # Model/training parameters
├── main.py             # Main pipeline trigger
├── app.py              # FastAPI app exposing /predict endpoint
├── Dockerfile          # Docker image definition (Python 3.12)
├── docker-compose.yml  # Compose file to run app locally
├── requirements.txt    # Python dependencies
├── setup.py            # For pip installation (editable mode)
├── research/           # Notebooks and EDA
│   └── research.ipynb
└── README.md           # You're here!
```

## Dataset
- This project focuses on intent classification for real estate and legal document analysis use cases. The model is trained to categorize incoming natural language emails or messages into one of eight specific intents relevant to commercial real estate workflows.

| Label | Intent Name                              | Description                                                                                   |
| ----- | ---------------------------------------- | --------------------------------------------------------------------------------------------- |
| `0`   | **Intent\_Lease\_Abstraction**           | Extract lease metadata like rent, term, parties, renewals, escalation clauses, etc.           |
| `1`   | **Intent\_Comparison\_LOI\_Lease**       | Compare LOI (Letter of Intent) with final lease clauses to identify changes or deviations.    |
| `2`   | **Intent\_Clause\_Protect**              | Identify missing, risky, or non-standard clauses such as indemnity, assignment rights, etc.   |
| `3`   | **Intent\_Company\_research**            | Conduct company background research, such as financial health, litigations, and credibility.  |
| `4`   | **Intent\_Transaction\_Date\_navigator** | Extract or summarize important transaction dates (e.g., closing, possession, notice periods). |
| `5`   | **Intent\_Amendment\_Abstraction**       | Identify and summarize what’s changed in lease amendments compared to original lease.         |
| `6`   | **Intent\_Sales\_Listings\_Comparison**  | Compare broker sales listings by price, square footage, cap rate, and other key metrics.      |
| `7`   | **Intent\_Lease\_Listings\_Comparison**  | Compare multiple lease listings to identify favorable terms and eliminate redundancy.         |


### 🧾 Dataset Format
Each row in the CSV files contains:

Column	Description
text	The input email/message written in natural language
label	Integer label (0–7) corresponding to one of the 8 intent classes

📌 Example
csv
```bash
email_text,label
"Can you pull together a schedule of important dates for the escrow process on the 125 King St deal?",4
"Compare the signed lease to the LOI we submitted last month.",1
"Please abstract the lease for the Johnson project. We need base rent, expiry date, and renewal options.",0
```

### 📊 Dataset Statistics
| Dataset Split    | Samples per Class | Total Samples |
| ---------------- | ----------------- | ------------- |
| **Training Set** | 1,000             | 8,000         |
| **Test Set**     | 100               | 800           |


The dataset is balanced, with an equal number(1000) of samples across all 8 intent categories, comes to total of 8000 training samples.

The test set contains novel samples, distinct from the training set, to evaluate generalization.(800 total samples)




## ⚙️ Pipeline Stages
The pipeline is modularized into the following stages:

1. **Data Ingestion**  
   - Load raw dataset
   - Split into train/test

2. **Data Validation**  
   - Schema checks (columns, types, missing values)

3. **Data Transformation**  
   - Tokenization using `AutoTokenizer` (from Hugging Face)
   - Encode `input_ids`, `attention_mask`, and `labels`

4. **Model Training**  
   - Transformer-based classifier (e.g., DistilBERT)
   - Uses `Trainer` class from Huggingface

5. **Model Evaluation**  
   - Generates metrics like accuracy, precision, recall, F1
   - Saves report as `.csv`

6. **Model Deployment (app.py)**  
   - FastAPI app that loads the trained model and serves predictions via REST API

## 🤖 Model
- **Architecture**: DistilBERT (`distilbert-base-uncased`)
- **Task**: Text Classification (Intent Detection)
- **Framework**: Hugging Face Transformers + PyTorch
- **Output**: Predicted intent label

## Run the API (Locally deployed FASTAPI application)
Test the trained model results by doing inference via FastAPI application

#### With Docker
```bash
docker-compose up --build
```

The API will be live at:
- http://localhost:8080/docs –> Swagger UI

## 📬 API Usage

### POST /predict

**Request:**
```json
{
  "email_text": "Could you do a background check on Wexford Corp before we proceed? I’m particularly interested in any public disputes or bankruptcies in the past 5 years."
}
```

**Response:**
```json
{
    "prediction": 3, 
    "class_label": "Intent_Company_research"
}
```

## 🧪 Test API
Use Swagger UI deployed locally at http://localhost:8080/docs


## 🚀 Running the end-to-end Application Pipeline
Data Ingestion, Data transformation, model training, and evaluation

### 1. Clone the Repository
```bash
git clone https://github.com/TejasGadi/intent-classification-using-distilbert.git
cd intent-classification-using-distilbert

# Download artifacts files from this source url (artifacts.zip) and paste inside root directory
https://drive.google.com/file/d/1h90FNbhz7huZiqKR5vFL69U5_sqkb6OF/view?usp=sharing"

```

### 2. Install Requirements (locally)
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Main Pipeline
```bash
python main.py
```
This triggers the end-to-end pipeline: data processing, training, and evaluation.


## 👨‍💻 Author
- Tejas Gadi
- 📧 tvgadi2003@gmail.com