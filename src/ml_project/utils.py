import os
import sys
from src.ml_project.exception import CustomException
from src.ml_project.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
database = os.getenv("database")

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydatabase = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        logging.info("Connection Established Successfully", mydatabase)
        df=pd.read_sql_query("Select * from students", mydatabase)
        print(df.head())

        return df



        pass
    except Exception as ex:
        raise CustomException(ex)
    