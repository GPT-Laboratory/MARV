import time
import os
import re
import requests
import pandas as pd
import streamlit as st
from typing import TypedDict, List, Dict, Literal
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langgraph.graph import StateGraph, END
from datetime import datetime
from qdrant_client.http.exceptions import ResponseHandlingException
import uuid


os.environ["STREAMLIT_WATCH_FILESYSTEM"] = "none"
# --- Config ---
API_URL = os.getenv("LLM_API_URL", "LLM_API_URL")
API_KEY = os.getenv("LLM_API_KEY", "LLM_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL", "llama3.1:latest")

REQ_PREFIX = "req_"
STD_PREFIX = "std_"
CHAT_PREFIX = "chat_"
VECTOR_SIZE = 384

# --- LLM Client ---
class LLMClient:
    def __init__(self):
        self.url = API_URL
        self.key = API_KEY
        self.model = MODEL_NAME
        self.session = self._create_session()

    def _create_session(self):
        session = requests.Session()
        retries = requests.adapters.Retry(total=3, backoff_factor=1)
        adapter = requests.adapters.HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def generate(self, prompt: str, max_tokens: int = 800, temperature: float = 0.7, timeout: int = 90) -> str:
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json"
        }
        payload = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        try:
            resp = self.session.post(self.url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response") or data.get("choices", [{}])[0].get("text", "").strip()
        except requests.exceptions.Timeout:
            return "[TIMEOUT] Response generation timed out."
        except requests.exceptions.RequestException as e:
            return f"[ERROR] {str(e)}"

# --- Embedding ---
class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        return self.model.encode(docs, normalize_embeddings=True).tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query], normalize_embeddings=True)[0].tolist()

embeddings = Embedder()

def create_qdrant_client_with_retry(max_retries=3, timeout=30):
    host = os.getenv("QDRANT_HOST", "localhost")
    port_candidates = [
        int(os.getenv("QDRANT_PORT", 6333)),
        6334 if int(os.getenv("QDRANT_PORT", 6333)) != 6334 else 6333
    ]
    for port in port_candidates:
        for attempt in range(max_retries):
            try:
                client = QdrantClient(host=host, port=port, timeout=timeout)
                client.get_collections()
                print(f"Connected to Qdrant at {host}:{port}")
                return client
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Qdrant connection attempt {attempt + 1} to {host}:{port} failed, retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print(f"Failed to connect to Qdrant at {host}:{port}: {e}")
    raise RuntimeError(f"Could not connect to Qdrant at {host} on ports {port_candidates}")

qdrant = create_qdrant_client_with_retry()
llm_client = LLMClient()

# --- Retriever ---
class RetrieverAgent:
    def __init__(self, embedder: Embedder, client: QdrantClient):
        self.embedder = embedder
        self.client = client

    def retrieve(self, query: str, collection: str, k: int = 5):
        qvec = self.embedder.embed_query(query)
        return self.client.search(collection_name=collection, query_vector=qvec, limit=k)

retriever = RetrieverAgent(embeddings, qdrant)

# --- LangGraph Setup ---
class AgentState(TypedDict):
    requirements: List[str]
    context: str
    conversation: List[Dict[str, str]]
    last_speaker: Literal["user", "developer", "Software Architect", "product_manager"]

def persona_prompt(role: str, context: str, history: List[Dict[str, str]], requirements: List[str]) -> str:
    persona_profiles = {
        "developer": "Alex, a senior software developer with 8+ years of experience. Focuses on technical feasibility, code quality, performance optimization, and implementation details. Asks probing questions about scalability, security, and maintainability.",
        "Software Architect": "Sarah, a software architect with deep system design expertise. Ensures architectural coherence, evaluates design patterns, considers long-term technical debt, and maintains traceability between requirements and implementation. Thinks holistically about system integration and technical dependencies.",
        "product_manager": "Michael, a product manager with strong business acumen. Prioritizes features based on user value, market impact, and ROI. Challenges technical decisions from a business perspective and ensures solutions align with user needs and company objectives."
    }
    chat_log = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in history)
    req_text = "\n".join(f"- {r}" for r in requirements)
    return f"""
You are {persona_profiles[role]}

Requirements under review:
{req_text}

Relevant Standards Context:
{context}

Conversation so far:
{chat_log}
Respond with your expert review of the requirements. You may agree, challenge, or refine previous points based on your expertise. Keep your tone professional, clear, and helpful.

⚠️ At the end of your response, append one validation table row like this:

| Requirement ID | Compliance Score (1-10) | Issue Type | Description | Suggested Fix | Matched Standard ID |
|----------------|--------------------------|------------|-------------|----------------|----------------------|
| REQ-001        | 7                        | Ambiguous  | Not clear how it will be measured. | Clarify measurement criteria. | STD-004 |

Make sure your table row is valid markdown, on its own line, and accurately reflects your evaluation.

Keep your response very short and concise and handoff the information to next agent . Summarize only critical issues, suggestions, or compliance flags. Avoid repeating full requirements unless necessary.
"""

# --- Standardize agent handoff in agent_node ---
def agent_node(role: str):
    agent_sequence = ["developer", "Software Architect", "product_manager"]
    def node(state: AgentState) -> AgentState:
        prompt = persona_prompt(role, state["context"], state["conversation"], state["requirements"])
        response = llm_client.generate(prompt, timeout=90)
        # Standardize handoff
        idx = agent_sequence.index(role)
        if idx < len(agent_sequence) - 1:
            next_agent = agent_sequence[idx + 1]
            handoff_sentence = f"Please hand this over to the {next_agent} for further review and analysis."
        else:
            handoff_sentence = "Validation cycle complete."
        # Remove any existing handoff-like sentence and append the uniform one
        lines = response.strip().splitlines()
        # Remove last line if it looks like a handoff
        if lines and ("hand" in lines[-1].lower() or "next agent" in lines[-1].lower() or "cycle" in lines[-1].lower()):
            lines = lines[:-1]
        lines.append(handoff_sentence)
        clean_response = "\n".join(lines)
        msg = {"role": role, "content": clean_response}
        return {**state, "conversation": state["conversation"] + [msg], "last_speaker": role}
    return node

def get_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("developer", agent_node("developer"))
    graph.add_node("Software Architect", agent_node("Software Architect"))
    graph.add_node("product_manager", agent_node("product_manager"))
    graph.set_entry_point("developer")
    graph.add_edge("developer", "Software Architect")
    graph.add_edge("Software Architect", "product_manager")
    return graph.compile()

agent_graph = get_agent_graph()

# --- Core Functions ---
def ensure_collection_exists(collection_name: str):
    try:
        existing = [c.name for c in qdrant.get_collections().collections]
        if collection_name not in existing:
            qdrant.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
    except Exception as e:
        st.warning(f"[Qdrant] Could not ensure collection '{collection_name}': {e}")

def store_conversations_to_qdrant(conversation: List[Dict[str, str]], collection_name: str):
    try:
        ensure_collection_exists(collection_name)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        points = [
            PointStruct(
                id=str(uuid.uuid4()), # Use UUID for Qdrant point ID
                vector=embeddings.embed_query(msg["content"]),
                payload={
                    "role": msg["role"],
                    "text": msg["content"],
                    "timestamp": current_time
                }
            )
            for i, msg in enumerate(conversation)
        ]
        batch_size = 50
        for i in range(0, len(points), batch_size):
            qdrant.upsert(collection_name=collection_name, points=points[i:i+batch_size])
        st.success(f"Stored conversation into '{collection_name}'")
    except Exception as e:
        st.warning(f"[Qdrant] Could not store conversation: {e}")

def store_full_session(conversation, collection_prefix="req_std_session"):
    session_text = "\n".join([f"{m['role']}: {m['content']}" for m in conversation])
    session_vector = embeddings.embed_query(session_text)
    session_id = str(uuid.uuid4())  # Use UUID for Qdrant point ID
    collection_name = f"{collection_prefix}"
    ensure_collection_exists(collection_name)
    point = PointStruct(
        id=session_id,  # This is now valid for Qdrant
        vector=session_vector,
        payload={
            "type": "validation_session",
            "session_id": session_id,
            "text": session_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )
    qdrant.upsert(collection_name=collection_name, points=[point])
    st.success(f"Full validation session stored as a single vector in '{collection_name}'!")
    return collection_name, session_id

class SummarizingAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def summarize(self, conversation_history):
        prompt = f"""
You are a senior requirements analyst. 
Based on the following agent conversation and validation table, write a *brief executive summary* for a project manager. 
- Focus on high-level critical issues, compliance gaps, or urgent actions.
- Avoid repeating the raw validation table or details for each requirement.
- Summarize in **at most 5 short bullet points**.
- Do NOT copy table markdown or IDs.
- Be clear, actionable, and professional.

Conversation and validation context:
{conversation_history}
"""
        return self.llm_client.generate(prompt, timeout=90)

# --- Utility for extracting markdown tables from text ---
def extract_table_blocks(text):
    # Regex to extract all markdown tables
    pattern = r'(\|.*?\|\n(\|.*?\|\n)+)'
    return re.findall(pattern, text)

def markdown_table_to_html(table_block):
    lines = [l for l in table_block.strip().splitlines() if l.startswith('|')]
    if len(lines) < 2:
        return ""
    header = [c.strip() for c in lines[0].split('|')[1:-1]]
    rows = []
    for l in lines[2:]:
        row = [c.strip() for c in l.split('|')[1:-1]]
        if len(row) == len(header):
            rows.append(row)
    html = "<table style='width:100%;border-collapse:collapse;margin-top:10px;'>"
    html += "<tr>" + "".join(f"<th style='border:1px solid #ccc;padding:4px;background:#f7fafd'>{h}</th>" for h in header) + "</tr>"
    for row in rows:
        html += "<tr>" + "".join(f"<td style='border:1px solid #eee;padding:4px;'>{c}</td>" for c in row) + "</tr>"
    html += "</table>"
    return html

# --- UI: Improved chat and summary display ---
def streamlit_agent_ui(selected_req: str, selected_stds: list):
    import re
    import pandas as pd
    st.markdown("## 🧠 Agent Conversation and Validation")
    session_key = generate_session_key(selected_req, selected_stds)
    col_chat, col_table = st.columns([2, 1.2])

    if session_key in st.session_state:
        table_rows = st.session_state[session_key]['table_rows']
        chat_history = st.session_state[session_key]['chat_history']
        summary_lines = st.session_state[session_key]['summary_lines']
        flagged_rows = st.session_state[session_key]['flagged_rows']
        final_summary = None
        # Extract and remove final summary from summary_lines if present
        filtered_summary_lines = []
        for line in summary_lines:
            if line.strip().startswith("### Final Summary") or line.strip().lower().startswith("final summary"):
                continue
            if line.strip():
                # If the line is the actual final summary (after the marker), save it
                if final_summary is None and ("Validation cycle complete" in line or line.strip().startswith("- ") is False and len(line.strip()) > 0):
                    final_summary = line.strip()
                else:
                    filtered_summary_lines.append(line)
        # If not found, just use the last line if it looks like a summary
        if not final_summary and summary_lines:
            final_summary = summary_lines[-1]

        def display_agent_message(role, content):
            # Split table and non-table parts
            table_blocks = extract_table_blocks(content)
            non_table_content = content
            for table_block, _ in table_blocks:
                non_table_content = non_table_content.replace(table_block, '')
            color_map = {
                'developer': '#c6f6d5',
                'Software Architect': '#e3f2fd',
                'product_manager': '#fff8e1',
                'Product_Manager': '#fff8e1'
            }
            icon_map = {
                'developer': '👨‍💻',
                'Software Architect': '🧑‍🏫',
                'product_manager': '📋',
                'Product_Manager': '📋'
            }
            color = color_map.get(role, '#f1f1f1')
            icon = icon_map.get(role, '💬')
            html = f"""
                <div style='margin: 10px 0; padding: 16px 20px; background: {color}; border-radius: 16px; border: 2px solid #ddd; font-size:1.09em;'>
                    <div style='font-weight:600;margin-bottom:7px'>{icon} {role}</div>
                    <div>{non_table_content.strip()}</div>
            """
            for table_block, _ in table_blocks:
                html += markdown_table_to_html(table_block)
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

        with col_chat:
            st.markdown("### 🤖 Agent Discussion")
            for role, content in chat_history:
                display_agent_message(role, content)
            if not chat_history:
                st.info("No agent discussion yet. Click Start Validation above.")

        with col_table:
            st.markdown("### 📝 Validation Table")
            if table_rows:
                df = pd.DataFrame(table_rows).drop_duplicates()
                if '__row_color__' in df.columns:
                    df = df.drop(columns=['__row_color__'])
                def highlight_rows(row):
                    try:
                        score = float(row['Compliance Score (1-10)'])
                        if score >= 7:
                            return ['background-color: #d4edda'] * len(row)
                        elif score >= 5:
                            return ['background-color: #fff3cd'] * len(row)
                        else:
                            return ['background-color: #f8d7da'] * len(row)
                    except (ValueError, TypeError):
                        return [''] * len(row)
                styled_df = df.style.apply(highlight_rows, axis=1)
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.warning("No validation results to display. If you expected results, check agent output or table extraction.")
            st.markdown("---")
            st.markdown("### 🚩 Flagged Requirements")
            flagged_req_ids = set()
            for row in flagged_rows:
                flagged_req_ids.add(row['Requirement ID'])
            if flagged_req_ids:
                for rid in flagged_req_ids:
                    st.markdown(f"- **{rid}** flagged")
            else:
                st.info("No flagged requirements. All requirements are compliant.")
            st.markdown("---")
            st.markdown("### 🔍 Executive Summary")
            # Only show concise summary (no tables, no verbose lines)
            if final_summary:
                st.markdown(f"{final_summary}")
            else:
                st.info("No major issues identified.")
    else:
        st.info("Click 'Start Validation' to begin.")

def run_agent_conversation(requirements: list, standard_collections: list):
    """
    Run the agent conversation for the given requirements and standard collections.
    Returns: conversation, table_rows, summary, context
    """
    context_hits = []
    for col in standard_collections:
        context_hits.extend(retriever.retrieve(" ".join(requirements), collection=col, k=5))
    context = "\n".join({hit.payload["text"] for hit in context_hits if "text" in hit.payload})

    initial_state = {
        "requirements": requirements,
        "context": context,
        "conversation": [{"role": "user", "content": "Can you validate these requirements?"}],
        "last_speaker": "user"
    }
    result = agent_graph.invoke(initial_state)
    conversation = result["conversation"]

    # Extract markdown table rows
    table_rows = []
    row_pattern = re.compile(r"\|\s*Requirement ID\s*\|.*?\|\s*Matched Standard ID\s*\|[\r\n]+((?:\|.*?\|[\r\n]+)+)")
    row_blocks = row_pattern.findall("\n".join(m["content"] for m in conversation if "content" in m))

    for block in row_blocks:
        lines = [line.strip() for line in block.strip().splitlines() if line.startswith("|") and not re.match(r"\|\s*-+", line)]
        for line in lines:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) == 6:
                row_dict = dict(zip([
                    "Requirement ID", "Compliance Score (1-10)", "Issue Type",
                    "Description", "Suggested Fix", "Matched Standard ID"
                ], cells))
                # Add color label based on score
                score_str = row_dict["Compliance Score (1-10)"].split(";")[0].strip()
                try:
                    score = int(score_str)
                    if score >= 7:
                        row_dict["__row_color__"] = "background-color: #d4edda"  # green
                    elif score >= 5:
                        row_dict["__row_color__"] = "background-color: #fff3cd"  # yellow
                    else:
                        row_dict["__row_color__"] = "background-color: #f8d7da"  # red
                except ValueError:
                    row_dict["__row_color__"] = ""
                table_rows.append(row_dict)

    # Build summary lines from table
    summary_lines = []
    for row in table_rows:
        score = row["Compliance Score (1-10)"].split(";")[0].strip()
        try:
            if score.isdigit() and int(score) < 7:
                summary_lines.append(f"- **{row['Requirement ID']}**: {row['Issue Type']} – {row['Description']}")
        except Exception:
            continue

    # Add final summary from the product manager (who acts as the summarizing agent)
    final_summary = next((msg["content"] for msg in reversed(conversation) if msg["role"] == "product_manager"), "")
    if final_summary:
        summary_lines.append("\n### Final Summary")
        summary_lines.append(final_summary)

    summary = "\n".join(summary_lines)
    return conversation, table_rows, summary, context

def group_table_rows_by_requirement(table_rows):
    """
    Group table rows by Requirement ID, combining all feedback and averaging compliance scores.
    """
    if not table_rows:
        return []
    grouped = {}
    for row in table_rows:
        req_id = row.get('Requirement ID', 'Unknown')
        if req_id not in grouped:
            grouped[req_id] = {
                'scores': [],
                'issue_types': [],
                'descriptions': [],
                'suggested_fixes': [],
                'matched_standards': []
            }
        score_str = row.get('Compliance Score (1-10)', '0').split(';')[0].strip()
        try:
            score = float(score_str)
            grouped[req_id]['scores'].append(score)
        except (ValueError, TypeError):
            pass
        issue_type = row.get('Issue Type', '').strip()
        description = row.get('Description', '').strip()
        suggested_fix = row.get('Suggested Fix', '').strip()
        matched_standard = row.get('Matched Standard ID', '').strip()
        if issue_type and issue_type not in grouped[req_id]['issue_types']:
            grouped[req_id]['issue_types'].append(issue_type)
        if description and description not in grouped[req_id]['descriptions']:
            grouped[req_id]['descriptions'].append(description)
        if suggested_fix and suggested_fix not in grouped[req_id]['suggested_fixes']:
            grouped[req_id]['suggested_fixes'].append(suggested_fix)
        if matched_standard and matched_standard not in grouped[req_id]['matched_standards']:
            grouped[req_id]['matched_standards'].append(matched_standard)
    final_rows = []
    for req_id, data in grouped.items():
        avg_score = round(sum(data['scores']) / len(data['scores']), 1) if data['scores'] else 0
        combined_issue_types = "; ".join(data['issue_types']) if data['issue_types'] else "No issues"
        combined_descriptions = "; ".join(data['descriptions']) if data['descriptions'] else "No description"
        combined_fixes = "; ".join(data['suggested_fixes']) if data['suggested_fixes'] else "No suggestions"
        combined_standards = "; ".join(data['matched_standards']) if data['matched_standards'] else "No standards"
        final_row = {
            'Requirement ID': req_id,
            'Compliance Score (1-10)': str(avg_score),
            'Issue Type': combined_issue_types,
            'Description': combined_descriptions,
            'Suggested Fix': combined_fixes,
            'Matched Standard ID': combined_standards
        }
        final_rows.append(final_row)
    return final_rows

def generate_session_key(selected_req, selected_stds):
    """
    Generates a unique session key for a given requirements collection and set of standard collections.
    """
    req_key = str(selected_req)
    std_key = "_".join(sorted(str(std) for std in selected_stds))
    return f"{req_key}__{std_key}"

def get_unique_flagged_and_total(grouped_table_rows):
    """
    Returns two sets: (flagged_req_ids, all_req_ids)
    flagged_req_ids: all unique Requirement IDs with score < 7
    all_req_ids: all unique Requirement IDs in the grouped table
    """
    flagged_req_ids = set()
    all_req_ids = set()
    for row in grouped_table_rows:
        req_ids = [r.strip() for r in row['Requirement ID'].split('&')]
        all_req_ids.update(req_ids)
        try:
            score = float(row['Compliance Score (1-10)'])
            if score < 7:
                flagged_req_ids.update(req_ids)
        except Exception:
            pass
    return flagged_req_ids, all_req_ids

__all__ = [
    "run_agent_conversation",
    "streamlit_agent_ui",
    "group_table_rows_by_requirement",
    "generate_session_key",
    "get_unique_flagged_and_total",
    # ... any other exports ...
]