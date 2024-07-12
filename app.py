

from src.mlproject.logger import logging 
from src.mlproject.exception import custom_exception
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_tranier import ModelTrainer
import sys



if __name__=="__main__":
    logging.info("The execution has started")

    try:
        data_ingest = DataIngestion()
        train_data_path,test_data_path = data_ingest.initiate_data_ingestion()
        logging.info(f"Train data path is {train_data_path}")
        data_transformations_obj = DataTransformation()
        train_arr,test_arr,_=data_transformations_obj.initiate_data_transormation(train_data_path,test_data_path)
        model_trainer = ModelTrainer()
        score = model_trainer.initiate_model_trainer(train_arr,test_arr)
        logging.info(f"Model score is {score}")

    except Exception as e:
        logging.info("Custom Exception")
        raise custom_exception(e,sys)