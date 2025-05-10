# Data Ingestion Config Schema/Data Class
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    train_csv_source_url: Path
    test_csv_source_url: Path
    local_train_csv_path: Path
    local_test_csv_path: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path
    local_train_csv_path: Path
    local_test_csv_path: Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    output_dir: Path
    logging_dir: Path
    eval_strategy: str
    learning_rate: float
    per_device_train_batch_size: int
    num_train_epochs: int
    weight_decay: float
    logging_steps: int

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_path: Path