from langchain_chroma import Chroma
from langchain.indexes import SQLRecordManager, index
from typing import List, Literal
from langchain_core.documents import Document
from loguru import logger
from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, GPT4AllEmbeddings
from config.config import CONFIG

class Indexer():
    def __init__(
            self,
            index_name: str,
            embedding_provider: Literal["cohere", "huggingface", "gpt4all"] = "cohere",
            embedding_model: str = "embed-english-v3.0",
            k: int = 3,
            score_treshold: float = 0.6,
        ):
        self.index_name = index_name
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model

        # Initialize embedding function based on provider
        if embedding_provider == "cohere":
            logger.debug(f"Using Cohere embeddings with model {embedding_model}")
            self.embedding_function = CohereEmbeddings(
                cohere_api_key=CONFIG.cohere_api_key,
                model=embedding_model
            )
        elif embedding_provider == "huggingface":
            logger.debug(f"Using HuggingFace embeddings with model {embedding_model}")
            model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
            encode_kwargs = {'normalize_embeddings': False}
            self.embedding_function = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        elif embedding_provider == "gpt4all":
            logger.debug(f"Using GPT4All embeddings with model {embedding_model}")
            self.embedding_function = GPT4AllEmbeddings(model_name=embedding_model)
        else:
            raise ValueError(f"Unsupported embedding provider: {embedding_provider}")

        self.vectorstore = Chroma(persist_directory=CONFIG.database_dir,
                                  embedding_function=self.embedding_function,
                                  collection_name=index_name)
        self.retriever = self.vectorstore.as_retriever(search_type="similarity_score_threshold",
                                                       search_kwargs={"score_threshold": score_treshold,
                                                                      "k": k},)

        self.namespace = f"chromadb/{index_name}"
        db_url = f"sqlite:///{CONFIG.database_dir}/record_manager_cache.sqlite3"
        self.record_manager = SQLRecordManager(self.namespace, db_url=db_url)
        self.record_manager.create_schema()

    def _clear(self):
        """Hacky helper method to clear content. See the `full` mode section to to understand why it works."""
        index([], self.record_manager, self.vectorstore, cleanup="full", source_id_key="source")
        
    def _index(self, 
               documents: List[Document],
               cleanup=None,
               source_id_key: str="source"):
        logger.debug(f"Indexing received documents - using {self.embedding_provider} with model {self.embedding_model}")
        result = index(
            documents,  
            self.record_manager,
            self.vectorstore,
            cleanup=cleanup,
            source_id_key=source_id_key,
        )
        
        logger.debug(f"Indexing done: {result}")
        return result

    def _get_doc_by_id(self,
                       _id: str):
        return self.vectorstore.get(where={"id": _id}).get('ids')
        
    def _delete_doc_from_index(self,
                               _id: str,
                               cleanup=None,
                               source_id_key: str="url"):
        """In order for the record manager to keep track of deleted documents, 
        we need to recreate the index with all documents minus the document to be deleted"""

        contents = self.vectorstore.get().get('documents')
        metadatas = self.vectorstore.get().get('metadatas')

        documents = [Document(page_content=doc, metadata=meta) 
                    for doc, meta in zip(contents, metadatas) 
                    if 'id' in meta.keys() and meta['id'] != _id]

        return index(
            documents,
            self.record_manager,
            self.vectorstore,
            cleanup="full",
            source_id_key=source_id_key,
        )
