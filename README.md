# Adarsh - Procurement & Policy Compliance RAG System

[![Tests](https://github.com/YOUR_ORG/Adarsh/workflows/Tests/badge.svg)](https://github.com/YOUR_ORG/Adarsh/actions?query=workflow:Tests)
[![Linting](https://github.com/YOUR_ORG/Adarsh/workflows/Linting/badge.svg)](https://github.com/YOUR_ORG/Adarsh/actions?query=workflow:Linting)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight, high-precision **Retrieval-Augmented Generation (RAG) system** for question-answering over procurement and Navy regulation policy documents. Built with BM25 lexical retrieval, metadata filtering, and Google Gemini API.

## 🎯 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation (5 minutes)

```bash
# Clone repository
git clone https://github.com/YOUR_ORG/Adarsh.git
cd Adarsh

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup API keys
cp .env.example .env
# Edit .env with your Gemini API key
```

### First Run

```bash
# Build the search index
python ingest.py

# Ask a question
python query.py --question "Under DPM 2025 Volume I, what authority level handles procurement approval?"

# Run evaluations
python eval.py

# Generate batch predictions
python run_pipeline.py
```

For detailed setup instructions, see [SETUP.md](SETUP.md).

---

## 🏗️ Architecture

### Why This Design?

#### 1. **BM25 Lexical Retrieval** (No Heavy Embeddings)
- ✅ **Zero-dependency fallback**: Pure Python implementation
- ✅ **Keyword specificity**: Perfect for rule references (e.g., "Rule 11", "DPM 2025 Volume I")
- ✅ **Transparency**: Understand exactly why documents are ranked

#### 2. **Metadata-Aware Filtering**
- Automatically detects document context in queries
- Filters search candidates before ranking
- Eliminates irrelevant matches for precision

#### 3. **Grounding-Focused Generation**
- Strict instructions to answer only from retrieved context
- JSON-structured outputs for reliable extraction
- Graceful refusal for out-of-domain questions

#### 4. **Citation Tracking**
- Every answer includes source document and section
- Automatic attribution for traceability

### System Flow

```
User Query
    ↓
Query Parser (detect document filters)
    ↓
BM25 Retriever (lexical ranking + metadata filter)
    ↓
Retrieved Context
    ↓
Gemini API (grounded generation)
    ↓
Structured JSON Response
    ├─ answer (grounded in context)
    ├─ source (document name)
    └─ section (exact section reference)
```

---

## 📁 File Structure

```
.
├── ingest.py              # Build BM25 index from metaData.csv
├── retrieve.py            # Query parser + BM25 retriever
├── query.py               # RAG interface with Gemini API
├── run_pipeline.py        # Batch predictions on test.csv
├── eval.py                # Evaluation metrics & grounding tests
├── Data/
│   ├── metaData.csv       # Pre-chunked document database
│   ├── sample_submission.csv
│   └── GenAI_Track3_Dataset/
│       └── Dataset/       # Source PDFs (DPM, Navy Regs)
├── index.pkl              # Serialized BM25 index (generated)
├── predictions.csv        # Output predictions (generated)
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── SETUP.md               # Detailed setup guide
├── CONTRIBUTING.md        # Contribution guidelines
└── LICENSE                # MIT License
```

---

## 🚀 Usage Examples

### Basic Query
```bash
python query.py --question "What is the threshold for Rule 11?"
```

**Output:**
```json
{
  "answer": "According to DPM 2025 Volume I, Rule 11 specifies...",
  "source": "DPM-2025-VOLUME-I.pdf",
  "section": "2.3 - Procurement Thresholds"
}
```

### Batch Predictions
```bash
# First 5 questions
python run_pipeline.py --limit 5

# All questions
python run_pipeline.py
```

### Evaluation
```bash
python eval.py
```

**Metrics:**
- Retrieval Recall@K (Recall@1, Recall@3, Recall@5)
- Citation Accuracy
- Refusal Accuracy

---

## 📊 Evaluation Metrics

The system measures three key aspects:

| Metric | Definition |
|--------|-----------|
| **Recall@K** | % of questions where ground-truth section is in top K results |
| **Citation Accuracy** | % of answers with correct source + section |
| **Refusal Accuracy** | % of unanswerable questions correctly identified |

See [README.md](README.md) for detailed architecture and metrics.

---

## 🔧 Configuration

### Environment Variables (`.env`)

```bash
# Required for LLM-powered queries
GEMINI_API_KEY=your_api_key_here

# Optional: Customize retrieval parameters
RETRIEVAL_TOP_K=5          # Top-K results to retrieve
MIN_RELEVANCE_SCORE=0.3    # Minimum relevance threshold
```

Get a free Gemini API key: https://makersuite.google.com/app/apikey

---

## 🛠️ Development

### Running Tests

```bash
# Run CI tests locally
python -m pytest test/ -v

# Run linting
flake8 . --max-line-length=100
```

### GitHub Actions

- **Tests**: Run on every push/PR (Python 3.8-3.11)
- **Linting**: Check code style with flake8, black, isort

See `.github/workflows/` for details.

---

## 📈 Future Improvements

- [ ] **Hybrid Retrieval**: Dense embeddings (all-MiniLM) + BM25
- [ ] **Query Expansion**: LLM-based acronym expansion
- [ ] **Context Windowing**: Retrieve surrounding chunks for continuity
- [ ] **Caching**: Redis for faster repeated queries
- [ ] **API Server**: FastAPI endpoint for production deployment

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Submission process
- Areas for contribution

Quick start:
```bash
git checkout -b feature/your-feature
# Make changes...
git commit -m "feat: add your feature"
git push origin feature/your-feature
# Create a Pull Request
```

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_ORG/Adarsh/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_ORG/Adarsh/discussions)
- **Documentation**: See [SETUP.md](SETUP.md) and [README.md](README.md)

---

## 🙏 Acknowledgments

Built for the GenAI Track 3 competition with focus on:
- Precision in legal/policy domain QA
- Transparency and citation accuracy
- Production-ready architecture

---

**Made with ❤️ for better policy compliance and procurement transparency**
