import json
import logging
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class AssessmentRetriever:
    def __init__(self, catalog_path: str = 'shl_product_catalog.json'):
        self.catalog_path = catalog_path
        self.items = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        
        self.load_and_index()

    def _format_item(self, item: dict) -> str:
        """Format an assessment item into a searchable string."""
        name = item.get('name', '')
        description = item.get('description', '')
        keys = " ".join(item.get('keys', []))
        job_levels = " ".join(item.get('job_levels', []))
        
        # Weighted formatting: Repeating name to give it more importance in TF-IDF
        return f"{name} {name} {keys} {job_levels} {description}"

    def load_and_index(self):
        logger.info(f"Loading catalog from {self.catalog_path}")
        try:
            with open(self.catalog_path, 'r', encoding='utf-8') as f:
                data = json.load(f, strict=False)
        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            return
            
        if isinstance(data, list):
            self.items = data
        else:
            self.items = list(data.values()) if isinstance(data, dict) else []
            
        if not self.items:
            logger.warning("No items loaded from catalog!")
            return

        logger.info(f"Loaded {len(self.items)} items. Initializing TF-IDF Index...")
        
        texts = [self._format_item(item) for item in self.items]
        
        # Fit and transform the texts to create the TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        logger.info("TF-IDF Indexing complete. Low-memory mode active.")

    def search(self, query: str, top_k: int = 15) -> List[dict]:
        if self.tfidf_matrix is None:
            return []
            
        # Transform the query using the same vectorizer
        query_vec = self.vectorizer.transform([query])
        
        # Calculate cosine similarity between query and all items
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get indices of top_k most similar items
        related_indices = similarities.argsort()[::-1][:top_k]
        
        results = []
        for idx in related_indices:
            # Only include if there's some similarity (> 0)
            if similarities[idx] > 0:
                results.append(self.items[idx])
        
        # If no similarity found, return first few items as fallback or empty
        return results

# Singleton instance
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
