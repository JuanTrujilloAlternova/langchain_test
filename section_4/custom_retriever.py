from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever


class RedundantVectorRetriever(VectorStoreRetriever):
    embeddings: Embeddings

    def __init__(self, vectorstore, embeddings, *args, **kwargs):
        super().__init__(vectorstore=vectorstore, embeddings=embeddings, **kwargs)
        self.embeddings = embeddings

    def get_relevant_documents(self, query, **kwargs) -> List[Document]:
        emb = self.embeddings.embed_query(query)
        return self.vectorstore.max_marginal_relevance_search_by_vector(
            emb
        )

    async def aget_relevant_documents(self, query, **kwargs) -> List[Document]:
        emb = self.embeddings.embed_query(query)
        return await self.db.amax_margin_relevance_search_by_vector(
            emb, k=1
        )

