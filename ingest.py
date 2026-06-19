import os
import pandas as pd
import pickle
import math
import re
from typing import List, Dict, Any
from retrieve import BM25Retriever

def main():
    print("Starting Ingestion...")
    metadata_path = os.path.join("Data", "metaData.csv")
    if not os.path.exists(metadata_path):
        print(f"Error: metadata file not found at {metadata_path}")
        return

    print(f"Reading chunks from {metadata_path}...")
    # Load metadata
    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Total chunks loaded: {len(df)}")
    
    # Standardize data structure
    documents = []
    for _, row in df.iterrows():
        # Ensure values are strings and not NaN
        chunk_id = str(row.get('chunk_id', ''))
        doc_name = str(row.get('document', ''))
        section = str(row.get('section', ''))
        text = str(row.get('text', ''))
        topic = str(row.get('topic', ''))

        documents.append({
            'chunk_id': chunk_id,
            'document': doc_name,
            'section': section,
            'text': text,
            'topic': topic
        })

    print("Fitting BM25 index...")
    retriever = BM25Retriever()
    retriever.fit(documents)

    # Save index to disk
    index_path = "index.pkl"
    print(f"Saving search index to {index_path}...")
    with open(index_path, "wb") as f:
        pickle.dump(retriever, f)
    
    print("Ingestion completed successfully!")

if __name__ == "__main__":
    main()
