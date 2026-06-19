# Setup Instructions

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Adarsh
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create a `.env` file by copying the template:
```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

You can get a free Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## Quick Start

### Index the Corpus
Build and save the search index:
```bash
python ingest.py
```

### Ask a Question
```bash
python query.py --question "Under DPM 2025 Volume I, what authority level generally handles procurement approval within delegated limits?"
```

### Run Evaluation
```bash
python eval.py
```

### Generate Full Batch Predictions
```bash
python run_pipeline.py
```

For quick testing on first 5 questions:
```bash
python run_pipeline.py --limit 5
```

## Project Overview

This is a **Retrieval-Augmented Generation (RAG) System** for question-answering over procurement and Navy regulation policy documents. It uses:

- **BM25 Lexical Retrieval**: Fast, keyword-specific retrieval without heavy embeddings
- **Metadata Filtering**: Smart document filtering for precise results
- **LLM Generation**: Google Gemini API for grounded answers
- **Citation Tracking**: Automatic source and section attribution

## Troubleshooting

### API Key Issues
- Ensure your `.env` file exists in the project root
- Verify your Gemini API key is valid and has not expired
- The system falls back to offline simulation mode if no API key is found

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Index File Not Found
Regenerate the index:
```bash
python ingest.py
```

## Support
For issues or questions, refer to the main [README.md](README.md) for architecture details and evaluation metrics.
