from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from settings import OPENAI_KEY

chat = ChatOpenAI(openai_api_key=OPENAI_KEY)
embeddings = OpenAIEmbeddings()

emb = embeddings.embed_query("What is the capital of France?")

