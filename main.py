from src.textClassifier.logging import logger
from src.textClassifier.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.textClassifier.pipeline.stage_2_data_transformation import DataTransformationPipeline
from src.textClassifier.pipeline.stage_3_model_trainer import ModelTrainerPipeline
from src.textClassifier.pipeline.stage_4_model_evaluation import ModelEvaluationPipeline

STAGE_NAME= "Data Ingestion Stage"

try:
    logger.info(f"stage: {STAGE_NAME} initiated")
    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"stage: {STAGE_NAME} completed")
except Exception as e:
    logger.exception(f"Error in main: {e}")
    raise e


STAGE_NAME= "Data Transformation Stage"

try:
    logger.info(f"stage: {STAGE_NAME} initiated")
    data_transformation_pipeline = DataTransformationPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    logger.info(f"stage: {STAGE_NAME} completed")
except Exception as e:
    logger.exception(f"Error in main: {e}")
    raise e

STAGE_NAME= "Model Trainer Stage"

try:
    logger.info(f"stage: {STAGE_NAME} initiated")
    model_trainer_pipeline = ModelTrainerPipeline()
    model_trainer_pipeline.initiate_model_trainer()
    logger.info(f"stage: {STAGE_NAME} completed")
except Exception as e:
    logger.exception(f"Error in main: {e}")
    raise e

STAGE_NAME= "Model Evaluation Stage"

try:
    logger.info(f"stage: {STAGE_NAME} initiated")
    model_evaluation_pipeline = ModelEvaluationPipeline()
    model_evaluation_pipeline.initiate_model_evaluation()
    logger.info(f"stage: {STAGE_NAME} completed")
except Exception as e:
    logger.exception(f"Error in main: {e}")
    raise e
