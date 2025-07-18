# MARV: Multi-Agent Requirement Validator

![MARV Logo](./assets/marv-logo.png) <!-- Replace with your logo path -->

MARV is an AI-powered software tool for automated validation of software requirements using a collaborative, multi-agent architecture. It leverages advanced LLMs and retrieval-augmented generation (RAG) to streamline requirements analysis, standards compliance, and team discussions—making SRS reviews faster, smarter, and more reliable.

---

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
- `requirements.txt` – Python dependencies
- `assets/` – (Put your logo and images here)

---

## 🤝 Contributing

1. Fork the repo and create a feature branch.
2. Submit a pull request with a clear description.
3. Open issues for suggestions or bugs.
4. See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## 📃 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Team & Contact

MARV is built by [Your Team/Organization].  
Questions or feedback? [Open an Issue](https://github.com/your-org/marv-multi-agent-requirement-validator/issues) or email us at youremail@example.com.

---

**MARV — Validate smarter. Build better.**
