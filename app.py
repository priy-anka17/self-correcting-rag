"""Self-Correcting RAG — Streamlit Application."""

import os
import sys
import time
import tempfile
import streamlit as st
import plotly.graph_objects as go

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from core.document_loader import load_and_chunk
from core.vector_store import (
    create_vector_store,
    load_vector_store,
    delete_collection,
)
from core.rag_pipeline import run_pipeline, PipelineStatus, IterationLog


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Self-Correcting RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global */
    .stApp { font-family: 'Inter', sans-serif; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: -10px;
        line-height: 1.6;
    }

    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin: 20px 0;
    }
    .feature-card {
        background: linear-gradient(145deg, #1e1e3a 0%, #1a1a2e 100%);
        border: 1px solid #2d2d5e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.25);
        border-color: #667eea;
    }
    .feature-icon { font-size: 2rem; margin-bottom: 8px; }
    .feature-title { font-weight: 700; font-size: 0.95rem; color: #e2e8f0; margin-bottom: 4px; }
    .feature-desc { font-size: 0.82rem; color: #94a3b8; line-height: 1.4; }

    /* Iteration cards */
    .iteration-card {
        border: 1px solid #2d2d5e;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 10px 0;
        background: linear-gradient(135deg, #1e1e3a 0%, #1a1a2e 100%);
        color: #e2e8f0;
    }

    /* Verdict badges */
    .pass-badge {
        background: linear-gradient(135deg, #065f46, #047857);
        color: #a7f3d0;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .fail-badge {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        color: #fecaca;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .partial-badge {
        background: linear-gradient(135deg, #78350f, #92400e);
        color: #fde68a;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Metric boxes */
    .metric-box {
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        background: #1a1a2e;
        border: 1px solid #2d2d5e;
        color: #e2e8f0;
    }

    /* Tech stack badges */
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 3px;
    }

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0e1117 100%);
    }

    .stExpander {
        border: 1px solid #2d2d5e;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ────────────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Enter your Groq API key (starts with gsk_)",
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    st.divider()

    st.markdown("### 📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "csv", "md"],
        accept_multiple_files=True,
        help="Upload PDF, TXT, CSV, or Markdown files",
    )

    if uploaded_files:
        if st.button("🔄 Process Documents", use_container_width=True):
            with st.spinner("Processing documents..."):
                # Save uploaded files to temp directory
                temp_dir = tempfile.mkdtemp()
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                # Load, chunk, and embed
                chunks = load_and_chunk(temp_dir)

                # Delete existing collection and create new
                delete_collection()
                st.session_state.vector_store = create_vector_store(chunks)
                st.session_state.documents_loaded = True
                st.success(f"✅ Processed {len(uploaded_files)} file(s) → {len(chunks)} chunks")

    # Load existing vector store if available
    if not st.session_state.documents_loaded:
        existing = load_vector_store()
        if existing:
            try:
                count = existing._collection.count()
                if count > 0:
                    st.session_state.vector_store = existing
                    st.session_state.documents_loaded = True
                    st.info(f"📚 Loaded existing knowledge base ({count} chunks)")
            except Exception:
                pass

    st.divider()
    st.markdown("### 🧪 Sample Data")
    if st.button("Load Sample Documents", use_container_width=True):
        sample_dir = os.path.join(os.path.dirname(__file__), "data", "sample")
        if os.path.exists(sample_dir) and os.listdir(sample_dir):
            with st.spinner("Loading sample documents..."):
                chunks = load_and_chunk(sample_dir)
                delete_collection()
                st.session_state.vector_store = create_vector_store(chunks)
                st.session_state.documents_loaded = True
                st.success(f"✅ Loaded sample data → {len(chunks)} chunks")
        else:
            st.warning("No sample documents found in data/sample/")

    st.divider()
    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        delete_collection()
        st.session_state.vector_store = None
        st.session_state.documents_loaded = False
        st.session_state.chat_history = []
        st.success("Knowledge base cleared")


# ── Main Content ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">Self-Correcting RAG</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Enterprise-grade Retrieval Augmented Generation with '
    "built-in hallucination detection & autonomous self-correction</p>",
    unsafe_allow_html=True,
)

# Tech stack badges
st.markdown(
    '<div style="margin: 5px 0 20px 0;">'
    '<span class="tech-badge">Groq LLM</span>'
    '<span class="tech-badge">LangChain</span>'
    '<span class="tech-badge">ChromaDB</span>'
    '<span class="tech-badge">Sentence Transformers</span>'
    '<span class="tech-badge">Streamlit</span>'
    '</div>',
    unsafe_allow_html=True,
)

# Feature cards
st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon">🛡️</div>
        <div class="feature-title">Hallucination Detection</div>
        <div class="feature-desc">Critic agent validates every answer against source documents before returning it</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🔄</div>
        <div class="feature-title">Auto Self-Correction</div>
        <div class="feature-desc">Failed answers trigger automatic query refinement & re-generation (up to 3x)</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Full Transparency</div>
        <div class="feature-desc">See every iteration, relevance score, and critic evaluation in real-time</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Architecture diagram
with st.expander("📐 How it works — Architecture", expanded=False):
    st.markdown("""
    ```
    ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
    │  User Query   │────▶│   Retriever      │────▶│   Generator      │
    └──────────────┘     │  (ChromaDB)      │     │  (Groq LLM)      │
                         └─────────────────┘     └────────┬─────────┘
                                  ▲                        │
                                  │                        ▼
                         ┌────────┴─────────┐    ┌──────────────────┐
                         │  Refined Query    │◀───│   Critic Agent    │
                         │  (if FAIL)        │    │  (Fact Checker)   │
                         └──────────────────┘    └──────────────────┘
                                                          │
                                                   ┌──────┴──────┐
                                                   │             │
                                                 PASS          FAIL
                                                   │             │
                                              Return Answer  Retry with
                                                            refined query
    ```

    **The Self-Correction Loop:**
    1. **Retrieve** — Fetch the most relevant document chunks from ChromaDB using semantic similarity
    2. **Generate** — Groq-hosted LLM produces an answer grounded strictly in the retrieved context
    3. **Critique** — A separate LLM call acts as a fact-checker, evaluating for hallucinations,
       contradictions, and missing information
    4. **Decide** — If the critic detects issues, the system automatically refines
       the search query and retries (up to 3 iterations)
    """)

st.divider()

# ── Chat Interface ───────────────────────────────────────────────────────────
if not st.session_state.documents_loaded:
    st.info("👈 Upload documents in the sidebar to get started, or load the sample data.")
else:
    # Display chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            if entry.get("details"):
                detail = entry["details"]
                status_emoji = {"success": "✅", "partial": "⚠️", "failed": "❌"}
                emoji = status_emoji.get(detail["status"], "")
                corrected = " (self-corrected)" if detail["was_corrected"] else ""
                st.caption(
                    f"{emoji} {detail['status'].title()}{corrected} "
                    f"• {detail['iterations']} iteration(s)"
                )

    # Chat input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        # Display user message
        with st.chat_message("user"):
            st.write(question)

        # Run pipeline
        with st.chat_message("assistant"):
            status_container = st.empty()
            progress_bar = st.progress(0)
            iteration_logs = []

            def on_iteration(log: IterationLog):
                iteration_logs.append(log)
                progress = min(log.iteration / 3, 1.0)
                progress_bar.progress(progress)
                verdict = log.critique.verdict.value
                emoji = "✅" if verdict == "PASS" else ("⚠️" if verdict == "PARTIAL" else "❌")
                status_container.markdown(
                    f"**Iteration {log.iteration}:** {emoji} {verdict} "
                    f"(confidence: {log.critique.confidence:.0%})"
                )

            start_time = time.time()
            result = run_pipeline(
                st.session_state.vector_store,
                question,
                on_iteration=on_iteration,
            )
            elapsed = time.time() - start_time

            # Clear progress indicators
            progress_bar.empty()
            status_container.empty()

            # Display answer
            st.markdown(result.final_answer)

            # ── Metrics Row ──────────────────────────────────────────
            st.divider()
            col1, col2, col3, col4 = st.columns(4)

            status_emoji = {
                PipelineStatus.SUCCESS: "✅",
                PipelineStatus.PARTIAL: "⚠️",
                PipelineStatus.FAILED: "❌",
            }

            with col1:
                st.metric("Status", f"{status_emoji[result.status]} {result.status.value.title()}")
            with col2:
                st.metric("Iterations", result.total_iterations)
            with col3:
                if result.final_critique:
                    st.metric("Confidence", f"{result.final_critique.confidence:.0%}")
            with col4:
                st.metric("Time", f"{elapsed:.1f}s")

            # ── Self-Correction Trace ────────────────────────────────
            if result.was_corrected:
                st.warning(
                    f"🔄 Self-correction activated! The system refined its answer "
                    f"over {result.total_iterations} iterations."
                )

            with st.expander("🔬 Detailed Pipeline Trace", expanded=result.was_corrected):
                for log in result.iterations:
                    verdict = log.critique.verdict.value
                    badge_class = verdict.lower() + "-badge"

                    st.markdown(
                        f'<div class="iteration-card">'
                        f'<strong>Iteration {log.iteration}</strong> '
                        f'<span class="{badge_class}">{verdict}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    tab1, tab2, tab3 = st.tabs(["📥 Retrieval", "📝 Answer", "🔍 Critique"])

                    with tab1:
                        st.markdown(f"**Query:** {log.query_used}")
                        st.markdown(f"**Avg Relevance Score:** {log.retrieval.avg_score:.3f}")
                        st.markdown(f"**Chunks Retrieved:** {len(log.retrieval.documents)}")
                        with st.expander("View Retrieved Chunks"):
                            for j, (doc, score) in enumerate(
                                zip(log.retrieval.documents, log.retrieval.scores)
                            ):
                                st.markdown(f"**Chunk {j+1}** (score: {score:.3f})")
                                st.text(doc.page_content[:500])
                                st.divider()

                    with tab2:
                        st.markdown(log.generation.answer)

                    with tab3:
                        st.markdown(f"**Verdict:** {log.critique.verdict.value}")
                        st.markdown(f"**Confidence:** {log.critique.confidence:.0%}")
                        st.markdown(f"**Reasoning:** {log.critique.reasoning}")

                        if log.critique.hallucinations:
                            st.error("**Hallucinations detected:**")
                            for h in log.critique.hallucinations:
                                st.markdown(f"- {h}")

                        if log.critique.contradictions:
                            st.error("**Contradictions found:**")
                            for c in log.critique.contradictions:
                                st.markdown(f"- {c}")

                        if log.critique.missing_info:
                            st.warning("**Missing information:**")
                            for m in log.critique.missing_info:
                                st.markdown(f"- {m}")

                        if log.critique.suggested_requery:
                            st.info(f"**Suggested re-query:** {log.critique.suggested_requery}")

            # ── Confidence Chart ─────────────────────────────────────
            if len(result.iterations) > 1:
                fig = go.Figure()
                iterations = list(range(1, len(result.iterations) + 1))
                confidences = [log.critique.confidence for log in result.iterations]
                relevances = [log.retrieval.avg_score for log in result.iterations]

                fig.add_trace(go.Scatter(
                    x=iterations, y=confidences,
                    mode="lines+markers",
                    name="Critic Confidence",
                    line=dict(color="#667eea", width=3),
                    marker=dict(size=10),
                ))
                fig.add_trace(go.Scatter(
                    x=iterations, y=relevances,
                    mode="lines+markers",
                    name="Retrieval Relevance",
                    line=dict(color="#764ba2", width=3),
                    marker=dict(size=10),
                ))
                fig.update_layout(
                    title="Self-Correction Progress",
                    xaxis_title="Iteration",
                    yaxis_title="Score",
                    yaxis=dict(range=[0, 1]),
                    template="plotly_white",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── Sources ──────────────────────────────────────────────
            if result.sources:
                with st.expander("📚 Sources"):
                    for source in result.sources:
                        st.markdown(f"- `{source}`")

            # Save to chat history
            st.session_state.chat_history.append({
                "question": question,
                "answer": result.final_answer,
                "details": {
                    "status": result.status.value,
                    "iterations": result.total_iterations,
                    "was_corrected": result.was_corrected,
                },
            })
