
import logging
from datetime import datetime
from pathlib import Path
import os

file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), "log", file_name)
os.makedirs(log_path, exist_ok=True)
log_file_path = os.path.join(log_path, file_name)

logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
