from src.textClassifier.constants import *
from src.textClassifier.utils.common import create_directories,read_yaml
from src.textClassifier.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig
from box import ConfigBox

# Create Configuration Manager: This has basic information required before staring any module/components

class ConfigurationManager:
    def __init__(self, config_path = CONFIG_FILE_PATH, params_path = PARAMS_FILE_PATH):
        self.config:ConfigBox = read_yaml(config_path)
        self.params:ConfigBox = read_yaml(params_path)


        # Create Artifacts Root dir
        create_directories([self.config.artifacts_root])
    
    # Read Data Ingestion Config
    def get_data_ingestion_config(self)->DataIngestionConfig:
        data_config = self.config.data_ingestion

        # create data root directory
        create_directories([data_config.root_dir])


        # This is the input that will go to the Data Ingestion c omponent
        data_ingestion_config = DataIngestionConfig(
            root_dir= data_config.root_dir,
            train_csv_source_url= data_config.train_csv_source_url,
            test_csv_source_url= data_config.test_csv_source_url,
            local_train_csv_path=data_config.local_train_csv_path,
            local_test_csv_path=data_config.local_test_csv_path
        )

        return data_ingestion_config

    def get_data_transformation_config(self)->DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name,
            local_train_csv_path= config.local_train_csv_path,
            local_test_csv_path= config.local_test_csv_path
        ) 

        return data_transformation_config
    
    def get_model_trainer_config(self)->ModelTrainerConfig:
        config = self.config.model_trainer

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_ckpt=config.model_ckpt,
            output_dir=self.params.TrainingArguments.output_dir,
            logging_dir=self.params.TrainingArguments.logging_dir,
            eval_strategy=self.params.TrainingArguments.eval_strategy,
            learning_rate=self.params.TrainingArguments.learning_rate,
            per_device_train_batch_size=self.params.TrainingArguments.per_device_train_batch_size,
            num_train_epochs=self.params.TrainingArguments.num_train_epochs,
            weight_decay=self.params.TrainingArguments.weight_decay,
            logging_steps=self.params.TrainingArguments.logging_steps
        )

        return model_trainer_config

    def get_model_evaluation_config(self)->ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            model_path = config.model_path,
            tokenizer_path = config.tokenizer_path,
            metric_file_path =  config.metric_file_path,
        )

        return model_evaluation_config
    
