from src.textClassifier.config.configuration import ConfigurationManager
from src.textClassifier.components.model_trainer import ModelTrainer
from src.textClassifier.logging import logger

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def initiate_model_trainer(self):   
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()

        model_trainer = ModelTrainer(model_trainer_config)

        model_trainer.train()