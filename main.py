try:
    import torch
    torch.classes.__path__ = []
except Exception:
    pass  # Safe fallback if torch is not installed yet

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ["WATCHDOG_OBSERVER_TIMEOUT"] = "1"
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"

import streamlit as st
import pandas as pd
import uuid
import time
from datetime import datetime
from agents_langgraph import (
    run_agent_conversation,
    store_conversations_to_qdrant,
    embeddings,
    qdrant,
    llm_client,
    ensure_collection_exists,
    REQ_PREFIX,
    STD_PREFIX,
    CHAT_PREFIX,
    streamlit_agent_ui,
    group_table_rows_by_requirement,
    generate_session_key,
    get_unique_flagged_and_total,
)
from qdrant_client.models import PointStruct

st.set_page_config(page_title="MARV", layout="wide")

st.title(":red[MARV] - A MULTI-AGENT APPROACH FOR VALIDATION OF SOFTWARE REQUIREMENTS")
st.info("➡️ Upload requirements ➡️ upload standards ➡️ run validation ➡️ review flagged issues & ask questions—all in four tabs above!")



# Initialize session state for validation metrics
if 'validation_start_time' not in st.session_state:
    st.session_state.validation_start_time = None
if 'total_requirements' not in st.session_state:
    st.session_state.total_requirements = 0
if 'flagged_requirements' not in st.session_state:
    st.session_state.flagged_requirements = 0
if 'validation_complete' not in st.session_state:
    st.session_state.validation_complete = False

tab1, tab2, tab3, tab4 = st.tabs(["Requirements", "Standards", "Validate", "Ask Questions"])


# --- Tab 1: Upload Requirements ---
with tab1:
    st.header("Upload Requirement Document")
    custom_name = st.text_input("Enter collection name for requirements")
    suffix = st.text_input("Add optional suffix (e.g. domain or project tag)")
    file = st.file_uploader("Upload .txt or .csv file", type=["txt", "csv"], key="req_file")

    if file and st.button("Upload Requirements"):
        texts = []
        if file.name.endswith(".txt"):
            texts = [line.strip() for line in file.read().decode("utf-8").splitlines() if line.strip()]
        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)
            if "Text" not in df.columns:
                st.error("CSV must have a 'Text' column.")
                st.stop()
            texts = df["Text"].dropna().tolist()

        name = custom_name.strip() if custom_name else uuid.uuid4().hex[:8]
        full_name = f"{REQ_PREFIX}{name}_{suffix.strip()}" if suffix else f"{REQ_PREFIX}{name}"

        vectors = embeddings.embed_documents(texts)
        ensure_collection_exists(full_name)
        points = [PointStruct(id=i, vector=vectors[i], payload={"text": texts[i], "rid": f"REQ-{i + 1:03}"}) for i in range(len(texts))]
        qdrant.upsert(collection_name=full_name, points=points)
        st.success(f"Uploaded {len(points)} requirements to '{full_name}'")
    

# --- Tab 2: Upload Standards ---
with tab2:
    st.header("Upload Standards Document")
    custom_name = st.text_input("Enter collection name for standards")
    suffix = st.text_input("Add optional suffix (e.g. regulation, team)")
    file = st.file_uploader("Upload .txt or .csv file", type=["txt", "csv"], key="std_file")

    if file and st.button("Upload Standards"):
        texts = []
        if file.name.endswith(".txt"):
            texts = [line.strip() for line in file.read().decode("utf-8").splitlines() if line.strip()]
        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)
            if "Text" not in df.columns:
                st.error("CSV must have a 'Text' column.")
                st.stop()
            texts = df["Text"].dropna().tolist()

        name = custom_name.strip() if custom_name else uuid.uuid4().hex[:8]
        full_name = f"{STD_PREFIX}{name}_{suffix.strip()}" if suffix else f"{STD_PREFIX}{name}"

        vectors = embeddings.embed_documents(texts)
        ensure_collection_exists(full_name)
        points = [PointStruct(id=i, vector=vectors[i], payload={"text": texts[i], "sid": f"STD-{i + 1:03}"}) for i in range(len(texts))]
        qdrant.upsert(collection_name=full_name, points=points)
        st.success(f"Uploaded {len(points)} standards to '{full_name}'")

def store_full_session(conversation, collection_prefix="req_std_session"):
    # Join messages into one text
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

# --- Tab 3: Validate Requirements ---
with tab3:
    st.header("Validate Requirements with MARV")
    
    # Scoreboard
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Requirements", st.session_state.total_requirements)
    with col2:
        st.metric("Flagged Requirements", st.session_state.flagged_requirements)
    with col3:
        if st.session_state.validation_start_time:
            elapsed_time = time.time() - st.session_state.validation_start_time
            st.metric("Time Elapsed", f"{elapsed_time:.2f} seconds")
    
    try:
        req_options = [c.name for c in qdrant.get_collections().collections if c.name.startswith(REQ_PREFIX)]
        std_options = [c.name for c in qdrant.get_collections().collections if c.name.startswith(STD_PREFIX)]
    except Exception as e:
        st.error(f"Cannot connect to Qdrant server: {str(e)}")
        st.info("Please ensure Qdrant server is running on the correct host and port (default: localhost:6333)")
        req_options = []
        std_options = []

    selected_req = st.selectbox("Select Requirement Collection", req_options)
    selected_stds = st.multiselect("Select Standard Collections", std_options)

    if selected_req and selected_stds:
        if st.button("Start Validation"):
            session_key = generate_session_key(selected_req, selected_stds)
            st.session_state.validation_start_time = time.time()
            st.session_state.validation_complete = False
            # Get all requirements from Qdrant
            hits = qdrant.scroll(collection_name=selected_req, limit=1000)[0]
            reqs = [{'Requirement ID': pt.payload.get('rid', f"REQ-{i}"), 'text': pt.payload['text']} for i, pt in enumerate(hits)]
            table_rows = []
            summary_lines = []
            flagged_rows = []
            chat_history = []
            with st.spinner("Running agent validation..."):
                try:
                    for req in reqs:
                        rid, text = req['Requirement ID'], req['text']
                        conversation, grouped_values, summary, _ = run_agent_conversation([f"{rid}: {text}"], selected_stds)
                        summary_lines.extend(summary.strip().splitlines())
                        for msg in conversation:
                            if msg['role'] == 'user':
                                continue
                            chat_history.append((msg['role'], msg['content']))
                        if isinstance(grouped_values, list):
                            table_rows.extend(grouped_values)
                    grouped_table_rows = group_table_rows_by_requirement(table_rows)
                    # Use new deduplication logic for metrics
                    flagged_req_ids, all_req_ids = get_unique_flagged_and_total(grouped_table_rows)
                    flagged_rows = [row for row in grouped_table_rows if row['Requirement ID'] in flagged_req_ids]
                    st.session_state.total_requirements = len(all_req_ids)
                    st.session_state.flagged_requirements = len(flagged_req_ids)
                    st.session_state[session_key] = {
                        'table_rows': grouped_table_rows,
                        'summary_lines': summary_lines,
                        'flagged_rows': flagged_rows,
                        'chat_history': chat_history,
                    }
                    st.session_state.validation_complete = True
                    st.success("Validation complete!")
                    # Store the full session transcript as a single vector in Qdrant
                    # Rebuild the full conversation as a list of dicts
                    full_conversation = []
                    for req in reqs:
                        rid, text = req['Requirement ID'], req['text']
                        conversation, _, _, _ = run_agent_conversation([f"{rid}: {text}"], selected_stds)
                        for msg in conversation:
                            if msg['role'] == 'user':
                                continue
                            full_conversation.append(msg)
                    store_full_session(full_conversation)
                except Exception as e:
                    import traceback
                    st.error(f"Error during validation: {str(e)}")
                    st.error(traceback.format_exc())
        # Always display results if present
        streamlit_agent_ui(selected_req, selected_stds)

# --- Tab 4: Ask Questions ---
with tab4:
    st.header("💬 Ask Questions About Your Validated Requirements")
    st.markdown("Get clarity, insights, and recommendations based on the **entire validation discussion**. Ask anything about the results, standards, flagged issues, or improvement ideas.")

    # Find all validation session transcripts
    session_collection = "req_std_session"
    try:
        sessions = qdrant.scroll(collection_name=session_collection, limit=1000)[0]
    except Exception:
        sessions = []
    if not sessions:
        st.warning("No completed validation sessions found. Please run validation in Tab 3 first.")
        st.stop()

    # Let user select which session (by timestamp or summary)
    options = [f"{pt.payload['timestamp']} (session_id: {pt.payload['session_id']})" for pt in sessions]
    selected_option = st.selectbox("Select a Validation Session", options, index=0)
    selected_idx = options.index(selected_option)
    selected_session = sessions[selected_idx]
    session_text = selected_session.payload['text']
    session_timestamp = selected_session.payload['timestamp']

    st.markdown("#### 📑 Validation Session Transcript")
    st.markdown(f"<div style='background:#f3f6fc;border-radius:10px;padding:16px 22px;font-size:1.12em;border:1.5px solid #d8dee9;margin-bottom:18px'>{session_text[:1000]}{'...' if len(session_text) > 1000 else ''}</div>", unsafe_allow_html=True)

    # Suggested questions (reuse your logic or keep simple)
    suggested = [
        "What are the most critical risks identified in this validation?",
        "Which requirements need urgent revision and why?",
        "How do our requirements align with selected standards?",
        "What are the top recommendations to improve quality?"
    ]
    st.markdown("#### 💡 Suggested Questions")
    qcols = st.columns(len(suggested))

    # --- Bulletproof session_state pattern for input reset ---
    if 'ask_tab4_input' not in st.session_state:
        st.session_state['ask_tab4_input'] = ''
    if 'ask_tab4_trigger' not in st.session_state:
        st.session_state['ask_tab4_trigger'] = False
    # If a trigger is set, set the value and rerun BEFORE the widget is rendered
    if st.session_state['ask_tab4_trigger']:
        st.session_state['ask_tab4_trigger'] = False
        st.experimental_rerun()

    # Render input widget using value from session_state
    question = st.text_input(
        "Ask anything about this validation session:",
        key="ask_tab4_input",
        value=st.session_state['ask_tab4_input']
    )
    session_key = f"tab4_chat_{selected_session.payload['session_id']}"
    if session_key not in st.session_state:
        st.session_state[session_key] = []
    for i, q in enumerate(suggested):
        if qcols[i].button(q, key=f"suggested_q_{i}"):
            st.session_state['ask_tab4_input'] = q
            st.session_state['ask_tab4_trigger'] = True
            st.experimental_rerun()
    if st.button("Get Answer", key="get_answer_tab4"):
        if not question.strip():
            st.info("Please enter a question first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    # Use the session transcript as context
                    prompt = f"""
You are an expert SRS validator. Using the following validation session transcript, answer the user's question clearly and specifically.

Session Transcript:
{session_text}

User's Question:
{question}

Be clear, concise, and actionable. Bullet points are fine.
"""
                    answer = llm_client.generate(prompt)
                except Exception:
                    answer = "[RAG/LLM is not available. Here is a fallback answer: Please check your validation results and flagged requirements above. If you need further help, contact your system administrator or try again later.]"
                st.session_state[session_key].append({'role': 'User', 'text': question})
                st.session_state[session_key].append({'role': 'MARV', 'text': answer})
                st.session_state['ask_tab4_input'] = ''
                st.session_state['ask_tab4_trigger'] = True
                st.experimental_rerun()
    st.markdown("---")
    st.markdown("#### 🗨️ Q&A Chat History")
    for msg in st.session_state[session_key]:
        color = "#d1e7dd" if msg['role'] == 'User' else "#fff3cd"
        align = "right" if msg['role'] == 'User' else "left"
        st.markdown(
            f"""
            <div style='margin: 10px 0; padding: 15px 20px; background: {color}; border-radius: 13px; border: 1.5px solid #d8dee9; text-align: {align}; max-width:80%;'>
                <b>{msg['role']}:</b> {msg['text']}
            </div>
            """,
            unsafe_allow_html=True
        )
