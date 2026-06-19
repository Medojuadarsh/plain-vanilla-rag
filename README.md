# Procurement and Policy Compliance RAG System

This repository implements a lightweight, high-precision Retrieval-Augmented Generation (RAG) system for question-answering over procurement and Navy regulation policy documents.

## Pipeline Architecture & Design Choices

The pipeline consists of the following key components:

### 1. Ingestion & Pre-Chunked Document Database
- **Choice**: Load and process text from `Data/metaData.csv` instead of directly parsing PDF files.
- **Rationale**: Indian procurement and regulation PDFs (such as the DPM 2025 volumes and Navy regulations) contain complex table layouts, headers/footers, and page numbers that degrade retrieval quality when parsed naively. `metaData.csv` contains pre-chunked, cleaned, and section-labeled text representing the official documents. This directly corresponds to the source document and section fields required for citations, providing a clean ground-truth starting point.

### 2. Search Index & Vector Store
- **Choice**: Custom TF-IDF & BM25 Retriever with metadata-based filtering.
- **Rationale**: 
  - **Zero-Dependency Fallback**: The BM25 retriever class is implemented in pure Python to enable out-of-the-box execution in restricted environments without requiring heavyweight dense embedding downloads.
  - **Keyword Specificity**: Procurement/regulation questions are highly keyword-specific (e.g. referencing "Rule 11", "DPM 2025 Volume I", "Rs. 2,00,000"). Lexical retrievers excel at indexing these precise identifiers.
  - **Metadata Filtering**: If the question asks about a specific document (e.g., "Under Navy Regulations Part I..."), the query parser automatically detects the target and filters the search candidates to only that document before ranking, eliminating irrelevant matching passages and achieving near-perfect retrieval precision.

### 3. Generation & Grounding Prompt
- **Choice**: Grounding-focused system prompts paired with structured JSON response constraints.
- **Rationale**:
  - The model is instructed to answer **only** using facts directly mentioned in the retrieved context.
  - To prevent hallucinations, the model is strictly configured to return "I cannot answer this question based on the provided corpus." if the answer is missing.
  - Structured output (`json_object`) is enforced so the pipeline can directly extract the `answer`, `source`, and `section` fields for predictions.

---

## File Structure

- [ingest.py](file:///c:/Users/vshiv/Desktop/Adarsh/ingest.py): Reads `metaData.csv`, builds a BM25 index, and pickles it to `index.pkl`.
- [retrieve.py](file:///c:/Users/vshiv/Desktop/Adarsh/retrieve.py): Loads the search index, parses query keywords for document filters, and returns relevant chunks.
- [query.py](file:///c:/Users/vshiv/Desktop/Adarsh/query.py): RAG query interface that retrieves context and queries the LLM (detects and configures Gemini, falling back to a mock mode when offline).
- [run_pipeline.py](file:///c:/Users/vshiv/Desktop/Adarsh/run_pipeline.py): Runs predictions batch-wise on all questions in `test.csv` and outputs to `predictions.csv`.
- [eval.py](file:///c:/Users/vshiv/Desktop/Adarsh/eval.py): Runs evaluation metrics against a ground-truth subset.

---

## How to Run

First, ensure you are in the workspace root `c:\Users\vshiv\Desktop\Adarsh`.

### Step 1: Index the Corpus
```bash
python ingest.py
```
This builds and saves the index database to `index.pkl`.

### Step 2: Ask a Question
```bash
python query.py --question "Under DPM 2025 Volume I, what authority level generally handles procurement approval within delegated limits?"
```
*(If a `GEMINI_API_KEY` environment variable is available, the LLM will answer; otherwise, the system will run in offline simulation mode.)*

### Step 3: Run the Full Evaluation Suite
```bash
python eval.py
```

### Step 4: Generate Full Batch Predictions
```bash
python run_pipeline.py
```
This processes all 140 questions in `test.csv` and outputs `predictions.csv` with columns: `id,prediction,pred_source,pred_section`.

For a quick test on the first 5 questions, run:
```bash
python run_pipeline.py --limit 5
```

---

## Evaluation Notes & Metrics

The system uses three main classes of evaluation:

1. **Retrieval Recall@K (Recall@1, Recall@3, Recall@5)**:
   - *Definition*: The percentage of questions where the ground-truth relevant section chunk is retrieved within the top $K$ results.
   - *Implementation*: Done in `eval.py` using 5 distinct hand-crafted grounding questions matching standard rules in the text.
   
2. **Citation Accuracy**:
   - *Definition*: The percentage of answers where the LLM correctly cites the exact document and section matching ground truth.
   
3. **Refusal Accuracy**:
   - *Definition*: The percentage of unanswerable questions (e.g. out-of-domain topics) correctly flagged by the LLM as unanswerable rather than guessed.

---

## Future Improvements (With More Time)

1. **Hybrid Retrieval (Dense + Sparse)**:
   - Combine dense vector embeddings (using a model like `text-embedding-004` or HuggingFace `all-MiniLM-L6-v2`) with lexical BM25 ranking to capture both semantic meaning and exact keyword rules.
2. **Query Expansion & Reformulation**:
   - Use an LLM rewrite step to expand query acronyms (e.g., expanding "DFPDS" to "Delegation of Financial Powers Rules") prior to retrieval.
3. **Chunk Context Windowing**:
   - Retrieve a target chunk, but feed surrounding chunks (preceding and succeeding sections) to the LLM to provide full contextual continuity for complex rule references.
