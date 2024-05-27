import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_HOST = os.getenv("DATABASE_HOST")
DATABASE_USER = os.getenv("DATABASE_USER")
DATABASE_PASSWD = os.getenv("DATABASE_PASSWD")
DATABASE_NAME = os.getenv("DATABASE_NAME")
DATABASE_PORT = os.getenv("DATABASE_PORT")
DATABASE_DRIVER = os.getenv("DATABASE_DRIVER", "psycopg")
