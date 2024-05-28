from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.tools import Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy

import settings

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    host=settings.DATABASE_HOST,
    user=settings.DATABASE_USER,
    password=settings.DATABASE_PASSWD,
    database=settings.DATABASE_NAME,
    port=settings.DATABASE_PORT,
    driver=settings.DATABASE_DRIVER
)
chat = ChatOpenAI(
    openai_api_key=settings.OPENAI_KEY,
    model_name="gpt-4o",
    temperature=0.0
)

def search_question_in_vector_store(question: str) -> str:
    embeddings = OpenAIEmbeddings()
    vector = PGVector(
        embeddings=embeddings,
        connection=CONNECTION_STRING,
        distance_strategy=DistanceStrategy.COSINE,
        collection_name="items"
    )
    retrieval = RetrievalQA.from_chain_type(
        retriever=vector.as_retriever(),
        llm=chat,
        chain_type="stuff",
    )
    return retrieval.invoke(question)


search_question_in_vector_store_tool = Tool(
    name="knowledge_base_search",
    func=search_question_in_vector_store,
    description="Search the knowledge base for a question",
)

search_from_google_tool = Tool(
    name="google_search",
    func=DuckDuckGoSearchRun().run,
    description="Search the web for a question",
)

def get_user_context(*args, **kwargs) -> dict:
    return {"name": "Juan Trujillo", "age": 25, "job": "Software Engineer"}


get_user_context_tool = Tool(
    name="get_user_context",
    func=get_user_context,
    description="Get the user context",
)
