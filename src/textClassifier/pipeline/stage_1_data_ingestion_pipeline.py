from src.textClassifier.config.configuration import ConfigurationManager
from src.textClassifier.components.data_ingestion import DataIngestion
from src.textClassifier.logging import logger



class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        # Instantiate Configuration Manager
        config_manager = ConfigurationManager()
        data_ingestion_config=config_manager.get_data_ingestion_config()

        data_ingestion = DataIngestion(config=data_ingestion_config)

        data_ingestion.download_dataset()
