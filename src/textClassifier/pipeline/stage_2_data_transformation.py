from src.textClassifier.config.configuration import ConfigurationManager
from src.textClassifier.components.data_transformation import DataTransformation
from src.textClassifier.logging import logger

class DataTransformationPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        # Instantiate and call the data transformation methods
        config = ConfigurationManager()

        data_transformation_config = config.get_data_transformation_config()

        data_transformation = DataTransformation(data_transformation_config)

        data_transformation.apply_transformation_to_dataset()