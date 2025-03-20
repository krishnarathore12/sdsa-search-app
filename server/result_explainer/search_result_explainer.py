import numpy as np
from langchain_cohere import CohereEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityExplainer:
    def __init__(self, search_index):
        self.model_type = 'cohere'
        self.model = CohereEmbeddings(
            model=search_index.embedding_model,
            cohere_api_key=search_index.embedding_function.cohere_api_key,
            client_name="search-app-server"
        )
        self.encoding_method = self.encode_cohere

    def encode_cohere(self, text):
        return np.array(self.model.embed_query(text))
    
    def similarity(self, emb1, emb2):
        return cosine_similarity([emb1], [emb2])[0][0]
    
    def explain_similarity(self, query, document, n_samples=5):
        """Simplified version that finds most similar terms using direct cosine similarity"""
        tokens = document.split()
        query_emb = self.encoding_method(query)
        token_embs = [self.encoding_method(token) for token in tokens]
        
        # Calculate similarities for each token
        similarities = [self.similarity(query_emb, token_emb) for token_emb in token_embs]
        token_importance = list(zip(tokens, similarities))
        
        # Return top N most similar tokens
        return [x for x in sorted(token_importance, key=lambda x: x[1], reverse=True) if x[1] > 0][:n_samples]
