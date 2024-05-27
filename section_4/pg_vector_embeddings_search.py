from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector, DistanceStrategy

import settings

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

results = db.similarity_search_with_score(
    "What is an interesting fact about Language",
    k=1
)

for data in results:
    print("\n")
    print(data[1])
    print(data[0].page_content)
