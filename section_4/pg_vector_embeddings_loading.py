from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters.character import CharacterTextSplitter

import settings

chat = ChatOpenAI(openai_api_key=settings.OPENAI_KEY)
loader = TextLoader("facts.txt")
text_splitter = CharacterTextSplitter(
    chunk_size=200,
    separator="\n",
    chunk_overlap=0
)
data = loader.load_and_split(text_splitter)
connection_string = PGVector.connection_string_from_db_params(
    driver=settings.DATABASE_DRIVER,
    host=settings.DATABASE_HOST,
    port=settings.DATABASE_PORT,
    user=settings.DATABASE_USER,
    password=settings.DATABASE_PASSWD,
    database=settings.DATABASE_NAME
)
embeddings = OpenAIEmbeddings()
db = PGVector.from_documents(
    embedding=embeddings,
    documents=data,
    connection=connection_string,
    collection_name="items",
    pre_delete_collection=True
)
