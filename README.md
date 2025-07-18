# MARV: Multi-Agent RAG Validator

A modern, multi-agent, RAG-powered tool for **validation of software requirements specifications**. MARV uses LLM agents, semantic search, and standards mapping to help you:

- Upload and vectorize requirements and standards
- Run multi-agent validation (developer, architect, product manager)
- See flagged requirements and compliance issues
- Ask questions about the entire validation session (chat with your results!)

---

## 🚀 Features

- **Requirements Upload:** Upload .txt or .csv files, vectorize, and store in Qdrant.
- **Standards Upload:** Upload .txt or .csv standards, vectorize, and store in Qdrant.
- **Multi-Agent Validation:** Developer, Architect, and Product Manager agents review requirements against standards, producing a validation table and flagged issues.
- **Session Transcript Storage:** Each validation run is stored as a single vector for semantic Q&A.
- **Ask Questions:** Chat with your validation session, get answers and recommendations from MARV.

---

## 🛠️ Setup & Installation

### 1. **Clone the repository**

```bash
git clone <your-repo-url>
cd MultiAgentValidator/MARV
```

### 2. **Install dependencies**

```bash
pip install -r requirements.txt
```

- Make sure you have Python 3.9+ and [Qdrant](https://qdrant.tech/documentation/quick-start/) running locally (default: `localhost:6333`).
- You may need to install [PyTorch](https://pytorch.org/get-started/locally/) for sentence-transformers.

### 3. **Set environment variables (optional)**

You can override the default LLM API, key, or model by setting:

- `LLM_API_URL`
- `LLM_API_KEY`
- `LLM_MODEL`

---

## ▶️ How to Run

```bash
streamlit run main.py
```

The app will open in your browser at [http://localhost:8501](http://localhost:8501)

---

## 🖥️ How to Use

### **Tab 1: Requirements**
- Upload a `.txt` or `.csv` file with your requirements (one per line or in a `Text` column).
- Name your collection and upload.

### **Tab 2: Standards**
- Upload a `.txt` or `.csv` file with standards (one per line or in a `Text` column).
- Name your collection and upload.

### **Tab 3: Validate**
- Select your requirements and standards collections.
- Click **Start Validation**.
- See:
  - Agent discussion (chat bubbles)
  - Validation table (with compliance scores, issues, suggestions)
  - Flagged requirements
  - Executive summary
- After validation, the full session transcript is stored for Q&A.

### **Tab 4: Ask Questions**
- Select a validation session (by timestamp).
- See the session transcript.
- Click a suggested question or type your own.
- Get answers from MARV, with chat history.

---

## ⚡ Troubleshooting

- **Qdrant not running?** Start Qdrant with Docker:
  ```bash
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
  ```
- **LLM API not responding?** Check your API URL and key, or use the default public endpoint.
- **Session state errors?** Make sure you are not modifying Streamlit session state after widgets are rendered (see code for bulletproof patterns).

---

## 📄 License

This project is for research and educational use. See LICENSE for details.

---

## 🤖 Authors

- [Your Name/Team]
- [Your Institution or Organization]

---

Enjoy using MARV for smarter, multi-agent requirements validation! 