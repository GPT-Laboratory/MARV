# MARV: Multi-Agent Requirement Validator

![MARV Logo](./assets/marvlogo.png)
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-1.30-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Qdrant-Vector%20DB-purple?logo=qdrant" />
  <img src="https://img.shields.io/badge/LLMs-LLaMA%203-green?logo=meta" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface" />
  <img src="https://img.shields.io/badge/RAG-Enabled-orange" />
</p>
MARV is an AI-powered  tool for automated validation of software requirements using a collaborative, multi-agent architecture. It leverages advanced LLMs and retrieval-augmented generation (RAG) to streamline requirements analysis, standards compliance, and team discussions, making SRS reviews faster, smarter, and more reliable.

---
![LandingPage](./assets/landingpage.png)
## 🚀 Features

- **Multi-Agent Validation:** Developer, Architect, Product Manager, and Summarizer agents collaborate to review each requirement from different perspectives.
- **RAG Workflow:** Seamlessly matches requirements to relevant standards using vector search (Qdrant) and sentence embeddings.
- **Interactive Streamlit UI:** Upload, validate, and explore flagged issues and feedback—all in an intuitive dashboard.
- **Session-Based Q&A:** Ask follow-up questions and receive context-aware answers based on full validation history.
- **Executive Summaries:** Instantly get concise, actionable summaries for decision makers.

---

## 🖥️ Quick Start

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/marv-multi-agent-requirement-validator.git
    cd marv-multi-agent-requirement-validator
    ```
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Start Qdrant (Vector DB):**
    ```bash
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
    ```
4. **Launch MARV:**
    ```bash
    streamlit run main.py
    ```

---

## 📂 Project Structure

- `main.py` – Streamlit app entry point
- `agents_langgraph.py` – Agent logic and core functions
- `requirements.txt` – Dependencies
- `assets/` – Project assets
- `input/` - Input files for project

---
## 🏗️ How to Use

- **Tab 1:** Upload your requirements (.txt or .csv)
- **Tab 2:** Upload standards for compliance checking
- **Tab 3:** Run validation — view agent discussion, flagged requirements, and summary
- **Tab 4:** Ask MARV questions about the full validation session!

---

## 🙋‍♂️ Team & Contact.  
Questions or feedback?
email us at harisrujan.chinnam@tuni.fi

---

**MARV — Validate smarter. Build better.**
