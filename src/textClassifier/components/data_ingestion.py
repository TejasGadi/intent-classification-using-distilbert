import os
import urllib.request as request
from src.textClassifier.logging import logger
from src.textClassifier.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_dataset(self):
        # Download Train Dataset
        if not os.path.exists(self.config.local_train_csv_path):
            request.urlretrieve(
                url=self.config.train_csv_source_url, 
                filename=self.config.local_train_csv_path
            )
            logger.info("✅ Train dataset downloaded.")
        else:
            logger.info("🟡 Train dataset already exists.")

        # Download Test Dataset
        if not os.path.exists(self.config.local_test_csv_path):
            request.urlretrieve(
                url=self.config.test_csv_source_url, 
                filename=self.config.local_test_csv_path
            )
            logger.info("✅ Test dataset downloaded.")
        else:
            logger.info("🟡 Test dataset already exists.")