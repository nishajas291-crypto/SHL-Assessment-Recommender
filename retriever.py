import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import logging

logger = logging.getLogger(__name__)

class AssessmentRetriever:
    def __init__(self, catalog_path: str = 'shl_product_catalog.json', model_name: str = 'all-MiniLM-L6-v2'):
        self.catalog_path = catalog_path
        self.model_name = model_name
        self.items = []
        self.index = None
        self.model = None
        
        self.load_and_index()

    def _format_item(self, item: dict) -> str:
        """Format an assessment item into a searchable string."""
        name = item.get('name', '')
        description = item.get('description', '')
        keys = ", ".join(item.get('keys', []))
        job_levels = ", ".join(item.get('job_levels', []))
        
        return f"Name: {name}\nKeys: {keys}\nJob Levels: {job_levels}\nDescription: {description}"

    def load_and_index(self):
        logger.info(f"Loading catalog from {self.catalog_path}")
        with open(self.catalog_path, 'r', encoding='utf-8') as f:
            data = json.load(f, strict=False)
            
        # The JSON has some malformed structure in real-world?
        # Ensure it's a list.
        if isinstance(data, list):
            self.items = data
        else:
            self.items = list(data.values()) if isinstance(data, dict) else []
            
        if not self.items:
            logger.warning("No items loaded from catalog!")
            return

        logger.info(f"Loaded {len(self.items)} items. Loading embedding model {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        
        texts = [self._format_item(item) for item in self.items]
        
        logger.info("Computing embeddings...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product with normalized vectors = Cosine Similarity
        self.index.add(embeddings)
        logger.info("Indexing complete.")

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        if not self.index or not self.model:
            return []
            
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.items):
                results.append(self.items[idx])
        return results

# Singleton instance to be used by the app
# Will be initialized in main.py on startup or lazily
retriever_instance = None

def get_retriever() -> AssessmentRetriever:
    global retriever_instance
    if retriever_instance is None:
        retriever_instance = AssessmentRetriever()
    return retriever_instance

if __name__ == '__main__':
    # Test
    logging.basicConfig(level=logging.INFO)
    ret = AssessmentRetriever()
    res = ret.search("I need a Rust coding assessment", top_k=2)
    for r in res:
        print(f"- {r.get('name')} | {r.get('link')}")
