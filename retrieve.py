import os
import pickle
import argparse
import math
import re
from typing import List, Dict, Any

# Custom BM25 implementation for zero-dependency keyword search
class BM25Retriever:
    def __init__(self, b=0.75, k1=1.5):
        self.b = b
        self.k1 = k1
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lengths = []
        self.doc_freqs = {}
        self.idf = {}
        self.documents = []
        self.doc_tokens = []

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase, keep alphanumeric, split by whitespace
        return re.findall(r'\b\w+\b', text.lower())

    def fit(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.corpus_size = len(documents)
        self.doc_tokens = [self._tokenize(doc['text']) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 1.0

        # Calculate document frequencies for words
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        # Calculate IDF for words
        for token, freq in self.doc_freqs.items():
            # Standard BM25 IDF formulation with smoothing
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5, doc_filter: str = None) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc in enumerate(self.documents):
            # Apply metadata filter on document name if specified
            if doc_filter and doc['document'] != doc_filter:
                continue

            score = 0.0
            doc_len = self.doc_lengths[idx]
            tokens = self.doc_tokens[idx]
            token_counts = {}
            for t in tokens:
                token_counts[t] = token_counts.get(t, 0) + 1

            for q_token in query_tokens:
                if q_token in token_counts:
                    tf = token_counts[q_token]
                    idf_val = self.idf.get(q_token, 0.0)
                    # BM25 tf-scaling formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += idf_val * (numerator / denominator)

            scores.append((score, doc))

        # Sort by score in descending order
        scores.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scores[:top_k]]

def detect_document_filter(query: str) -> str:
    """
    Detects which document the query refers to, to enable metadata filtering.
    """
    query_lower = query.lower()
    
    # Check DPM 2025 Vol I vs II
    if "dpm 2025 volume i" in query_lower or "dpm 2025 vol i" in query_lower or "dpm 2025 volume 1" in query_lower:
        return "DPM-2025-VOLUME-I.pdf"
    if "dpm 2025 volume ii" in query_lower or "dpm 2025 vol ii" in query_lower or "dpm 2025 volume 2" in query_lower:
        return "DPM-2025-VOLUME-II.pdf"
        
    # Check DFPDS
    if "dfpds" in query_lower or "delegation of financial powers" in query_lower or "dfpr" in query_lower:
        return "Delegation_of_Financial_Powers_Rules_2024_Booklet.pdf"
        
    # Check Navy Regulations
    if "navy regulations part i" in query_lower or "regsnavyi" in query_lower or "navy reg part i" in query_lower or "regs navy i" in query_lower:
        return "RegsNavyI.pdf"
    if "navy regulations part ii" in query_lower or "regsnavyii" in query_lower or "navy reg part ii" in query_lower or "regs navy ii" in query_lower:
        return "RegsNavyII.pdf"
    if "navy regulations part iii" in query_lower or "regsnavyiii" in query_lower or "navy reg part iii" in query_lower or "regs navy iii" in query_lower:
        return "RegsNavyIII.pdf"
    if "navy regulations part iv" in query_lower or "regsnavyiv" in query_lower or "navy reg part iv" in query_lower or "regs navy iv" in query_lower:
        return "RegsNavyIV.pdf"
        
    return None

def retrieve_context(query: str, index_path: str = "index.pkl", top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Loads the index and retrieves top_k relevant chunks.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file {index_path} not found. Please run ingest.py first.")
        
    with open(index_path, "rb") as f:
        retriever = pickle.load(f)
        
    doc_filter = detect_document_filter(query)
    if doc_filter:
        print(f"[Filter] Filtering search to document: {doc_filter}")
    else:
        print("[Filter] No specific document filter detected. Searching whole corpus.")
        
    return retriever.search(query, top_k=top_k, doc_filter=doc_filter)

def main():
    parser = argparse.ArgumentParser(description="Query the RAG retrieval component")
    parser.add_argument("--query", type=str, required=True, help="Query string")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to return")
    parser.add_argument("--index", type=str, default="index.pkl", help="Path to index file")
    
    args = parser.parse_args()
    
    try:
        results = retrieve_context(args.query, index_path=args.index, top_k=args.top_k)
        print(f"\nRetrieved {len(results)} chunks for query: '{args.query}'\n")
        for i, res in enumerate(results):
            print(f"--- Chunk {i+1} ---")
            print(f"ID: {res['chunk_id']}")
            print(f"Document: {res['document']}")
            print(f"Section: {res['section']}")
            print(f"Topic: {res['topic']}")
            print(f"Text:\n{res['text'][:200]}...")
            print("-" * 20 + "\n")
    except Exception as e:
        print(f"Error during retrieval: {e}")

if __name__ == "__main__":
    main()
