import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, documents):
        """
        documents: list of strings  (news chunks)
        """
        self.documents = documents
        
        # ---- BM25 ----
        tokenized_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # ---- Semantic Embeddings ----
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(documents, convert_to_numpy=True)
    
    def bm25_search(self, query):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        return np.array(scores)
    
    def semantic_search(self, query):
        query_emb = self.model.encode([query], convert_to_numpy=True)[0]
        
        # cosine similarity
        scores = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        return scores
    
    def hybrid_search(self, query, top_k=5, alpha=0.5):
        """
        alpha controls weighting:
        alpha = 0.5 → equal weight
        alpha → 1 → more semantic
        alpha → 0 → more BM25
        """
        bm25_scores = self.bm25_search(query)
        semantic_scores = self.semantic_search(query)
        
        # Normalize scores
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
        
        # Combine
        hybrid_scores = alpha * semantic_norm + (1 - alpha) * bm25_norm
        
        # Get top results
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = [(self.documents[i], hybrid_scores[i]) for i in top_indices]
        return results