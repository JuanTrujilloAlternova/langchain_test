import langchain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

import settings
from section_4.custom_prompt_template import CUSTOM_PROMPT
from section_4.custom_retriever import RedundantVectorRetriever


chat = ChatOpenAI(openai_api_key=settings.OPENAI_KEY, model_name="gpt-4o")
connection_string = PGVector.connection_string_from_db_params(
    driver=settings.DATABASE_DRIVER,
    host=settings.DATABASE_HOST,
    port=settings.DATABASE_PORT,
    user=settings.DATABASE_USER,
    password=settings.DATABASE_PASSWD,
    database=settings.DATABASE_NAME
)
embeddings = OpenAIEmbeddings()
db = PGVector(
    embeddings=embeddings,
    connection=connection_string,
    collection_name="items",
)

retriever = RedundantVectorRetriever(
    vectorstore=db,
    embeddings=embeddings,
)

chain = RetrievalQA.from_chain_type(
    retriever=retriever,
    llm=chat,
    chain_type="stuff",
    chain_type_kwargs={"prompt": CUSTOM_PROMPT}
)

result = chain.invoke("What is a relevant fact about English?")

print(result)
