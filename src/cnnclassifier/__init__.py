import os
import sys
import logging

logging_str = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"


log_dir="logs"
log_filepath=os.path.join(log_dir, "cnnclassifier.log")
os.makedirs(log_dir, exist_ok=True) 



logging.basicConfig(level=logging.DEBUG, format=logging_str
                    handlers=[
                        logging.FileHandler(log_filepath),
                        logging.StreamHandler(sys.stdout)
                    ])
logger=logging.getLogger("cnnclassifier")


logging.info("This is a test log message.")
