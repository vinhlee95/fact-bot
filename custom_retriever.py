from typing import Any, Dict, List, Optional
from langchain.schema.retriever import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun, Callbacks
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma


class CustomRetriever(BaseRetriever):
  embeddings: Embeddings
  chroma: Chroma

  def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
    embedded_query = self.embeddings.embed_query(query)

    return self.chroma.max_marginal_relevance_search_by_vector(
      embedding=embedded_query,
      lambda_mult=0.7,
    )
  
  async def aget_relevant_documents(self, query: str, *, callbacks: Callbacks = None, tags: List[str] | None = None, metadata: Dict[str, Any] | None = None, run_name: str | None = None, **kwargs: Any) -> List[Document]:
    return await super().aget_relevant_documents(query, callbacks=callbacks, tags=tags, metadata=metadata, run_name=run_name, **kwargs)