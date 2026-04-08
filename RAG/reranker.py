from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents):
        """
        documents: list of (doc_text, score) from hybrid retriever
        """
        texts = [doc for doc, _ in documents]
        
        pairs = [(query, doc) for doc in texts]
        scores = self.model.predict(pairs)

        reranked = list(zip(texts, scores))
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked