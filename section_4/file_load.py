from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings

from settings import OPENAI_KEY

chat = ChatOpenAI(openai_api_key=OPENAI_KEY)
loader = TextLoader("facts.txt")
data = loader.load()
embeddings = OpenAIEmbeddings()

print(data)
