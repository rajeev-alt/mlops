

from src.mlproject.logger import logging 
from src.mlproject.exception import custom_exception
from src.mlproject.components.data_ingestion import DataIngestion
import sys



if __name__=="__main__":
    logging.info("The execution has started")

    try:
        data_ingest = DataIngestion()
        train_data_path,test_data_path = data_ingest.initiate_data_ingestion()
    except Exception as e:
        logging.info("Custom Exception")
        raise custom_exception(e,sys)