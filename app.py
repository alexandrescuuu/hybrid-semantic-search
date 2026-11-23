import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from core import DEFAULT_DOCUMENTS, split_documents, HybridSearchEngine

st.set_page_config(
    page_title="Hybrid Semantic & Lexical Search Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Hybrid Semantic & Lexical Search Engine")
st.markdown(
    "Information Retrieval engine blending **Sparse Keyword Matching (TF-IDF)** with "
    "**Dense Semantic Embeddings (LSA + FastText OOV)** via tunable score fusion (α)."
)


# Sidebar controls
st.sidebar.header("Knowledge Base & Retrieval Settings")
raw_docs = st.sidebar.text_area(
    "Document Corpus (1 document per line)",
    value=DEFAULT_DOCUMENTS,
    height=260,
)
alpha = st.sidebar.slider(
    "Fusion Weight α (0 = Pure Semantic, 1 = Pure Lexical)",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
)
top_k = st.sidebar.slider("Top-K Results", min_value=2, max_value=10, value=5)

docs = split_documents(raw_docs)
if len(docs) < 3:
    st.error("Please provide at least 3 non-empty lines/documents in the corpus.")
    st.stop()

engine = HybridSearchEngine(docs)

# Search Bar
st.subheader("Search Query")
sample_queries = (
    "Custom query...",
    "undecidable programs and infinite loops",
    "handling unseen misspelled words with subwords",
    "19th century mechanical computing Bernoulli",
    "minimizing error with gradient updates",
)
preset_q = st.selectbox("Try a conceptual query (with zero exact keyword overlap) or write your own:", sample_queries)
default_q = "undecidable programs and infinite loops" if preset_q == "Custom query..." else preset_q
query = st.text_input("Enter your search query:", value=default_q)

if query.strip():
    results_df = engine.search(query=query, alpha=alpha, top_k=top_k)

    col1, col2, col3 = st.columns(3)
    top_row = results_df.iloc[0]
    col1.metric("Top Match", top_row["Doc ID"])
    col2.metric("Hybrid Relevance Score", f"{top_row['Hybrid Score']:.4f}")
    mode_label = "Balanced Hybrid" if 0.2 <= alpha <= 0.8 else ("Lexical-Heavy" if alpha > 0.8 else "Semantic-Heavy")
    col3.metric("Retrieval Mode", mode_label)

    st.dataframe(results_df, use_container_width=True)

    # Bar chart comparing Lexical vs Semantic contribution
    st.subheader("Score Breakdown: Lexical vs. Semantic Retrieval")
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=120)
    x = range(len(results_df))
    width = 0.35
    ax.bar([i - width / 2 for i in x], results_df["Lexical (TF-IDF)"], width=width, label="Lexical (TF-IDF)", color="#1f77b4")
    ax.bar([i + width / 2 for i in x], results_df["Semantic (LSA+FT)"], width=width, label="Semantic (LSA + FastText)", color="#FF4B4B")
    ax.set_xticks(list(x))
    ax.set_xticklabels(results_df["Doc ID"])
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Score Decomposition for Query: '{query}'")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
