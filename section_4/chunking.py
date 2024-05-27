from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

from settings import OPENAI_KEY

chat = ChatOpenAI(openai_api_key=OPENAI_KEY)
loader = TextLoader("facts.txt")
text_splitter = CharacterTextSplitter(
    chunk_size=100,
    separator="\n",
    chunk_overlap=0
)
data = loader.load_and_split(text_splitter)

for chunk in data:
    print(chunk)
