"""
Test search functionality without UI
"""
import sys
import yaml
from pathlib import Path

sys.path.insert(0, '.')

from src.embeddings.embedding_model import BGEM3Embedder
from src.search.vector_store import QdrantVectorStore
from src.search.bm25_search import BM25SearchEngine
from src.search.hybrid_search import HybridSearchEngine

print("=" * 60)
print("AI Agent Search Test")
print("=" * 60)

# Load config
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Initialize components
print("\n1. Initializing embedding model...")
embedder = BGEM3Embedder(
    model_name=config['embedding']['model_name'],
    device=config['embedding']['device'],
    use_fp16=config['embedding']['use_fp16']
)
print("✅ Embedder initialized")

print("\n2. Connecting to vector store...")
vector_store = QdrantVectorStore(
    storage_path=config['data']['qdrant_storage'],
    collection_name=config['qdrant']['collection_name']
)
print(f"✅ Vector store connected ({vector_store.count_documents()} documents)")

print("\n3. Loading BM25 index...")
bm25_engine = BM25SearchEngine(use_korean_tokenizer=True)
bm25_index_path = Path(config['data']['cache_dir']) / 'bm25_index.pkl'
if bm25_index_path.exists():
    bm25_engine.load_index(str(bm25_index_path))
    print("✅ BM25 index loaded")

print("\n4. Initializing hybrid search...")
hybrid_engine = HybridSearchEngine(
    vector_store=vector_store,
    bm25_engine=bm25_engine,
    bm25_weight=config['search']['bm25_weight'],
    vector_weight=config['search']['semantic_weight']
)
print("✅ Hybrid search engine ready")

# Test search
print("\n" + "=" * 60)
print("Testing search...")
print("=" * 60)

test_query = "2024년 매출 보고서"
print(f"\nQuery: {test_query}")

# Generate query embedding
print("Generating query embedding...")
query_embedding = embedder.encode_queries(test_query)

# Search
print("Searching...")
results = hybrid_engine.search(
    query=test_query,
    query_embedding=query_embedding,
    top_k=5
)

# Display results
print(f"\n✅ Found {len(results)} results:\n")
for i, result in enumerate(results, 1):
    print(f"{i}. Score: {result['score']:.4f}")
    print(f"   Keys: {list(result.keys())}")
    print(f"   Data: {result}")
    print()

print("=" * 60)
print("✅ Search test completed successfully!")
print("=" * 60)
