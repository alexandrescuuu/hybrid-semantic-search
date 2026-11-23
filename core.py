import re
import numpy as np
import pandas as pd
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_DOCUMENTS = """
Turing Machines provide a formal mathematical model of computation using an infinite tape and a finite state transition table.
The Halting Problem proves that no general algorithm can decide whether an arbitrary program finishes or runs forever.
Ada Lovelace wrote Note G in 1843, describing an algorithm for Charles Babbage's Analytical Engine to compute Bernoulli numbers.
Word2Vec learns dense vector representations of words by predicting surrounding context tokens using Skip-gram or CBOW architectures.
FastText extends Word2Vec by representing each word as a bag of character n-grams, enabling embeddings for out-of-vocabulary words.
Latent Semantic Analysis applies Truncated Singular Value Decomposition to a TF-IDF matrix to uncover hidden conceptual relationships.
Byte Pair Encoding iteratively merges the most frequent adjacent symbol pairs to construct a compact subword vocabulary for LLMs.
Gradient Descent optimizes differentiable loss functions by updating model parameters in the opposite direction of the gradient.
Logistic Regression models binary class probabilities via the sigmoid activation function and minimizes cross-entropy log-loss.
Hybrid Information Retrieval combines sparse lexical matching (TF-IDF) with dense semantic embeddings to improve search recall.
""".strip()


def split_documents(raw_text: str) -> list[str]:
    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower())


class HybridSearchEngine:
    """Combines Sparse Lexical Retrieval (TF-IDF) with Dense Semantic Retrieval (LSA + FastText)."""

    def __init__(self, documents: list[str], n_components: int = 5):
        self.documents = documents
        self.tokenized_docs = [tokenize(d) for d in documents]

        # 1. Sparse Lexical Index (TF-IDF)
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(documents)

        # 2. Latent Semantic Index (LSA / Truncated SVD)
        k = min(n_components, max(1, min(self.tfidf_matrix.shape) - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=42)
        self.lsa_matrix = self.svd.fit_transform(self.tfidf_matrix)

        # 3. Neural Subword Embeddings (FastText)
        self.ft = FastText(
            sentences=self.tokenized_docs * 30,
            vector_size=32,
            window=4,
            min_count=1,
            min_n=2,
            max_n=4,
            epochs=120,
            seed=42,
        )
        self.doc_ft_vectors = np.vstack([self._embed_tokens(toks) for toks in self.tokenized_docs])

    def _embed_tokens(self, tokens: list[str]) -> np.ndarray:
        if not tokens:
            return np.zeros(self.ft.vector_size)
        vecs = [self.ft.wv[t] for t in tokens]
        mean_vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        return mean_vec / norm if norm > 0 else mean_vec

    def search(self, query: str, alpha: float = 0.5, top_k: int = 5) -> pd.DataFrame:
        """
        alpha = 1.0 -> Pure Lexical Search (TF-IDF)
        alpha = 0.0 -> Pure Semantic Search (50% LSA + 50% FastText)
        """
        q_tfidf = self.tfidf.transform((query,))
        lexical_scores = cosine_similarity(q_tfidf, self.tfidf_matrix).ravel()

        q_lsa = self.svd.transform(q_tfidf)
        lsa_scores = cosine_similarity(q_lsa, self.lsa_matrix).ravel()

        q_ft = self._embed_tokens(tokenize(query)).reshape(1, -1)
        ft_scores = cosine_similarity(q_ft, self.doc_ft_vectors).ravel()

        semantic_scores = 0.5 * np.clip(lsa_scores, 0, 1) + 0.5 * np.clip(ft_scores, 0, 1)
        hybrid_scores = alpha * lexical_scores + (1.0 - alpha) * semantic_scores

        results = []
        for idx, doc in enumerate(self.documents):
            results.append({
                "Doc ID": f"D{idx + 1}",
                "Hybrid Score": round(float(hybrid_scores.item(idx)), 4),
                "Lexical (TF-IDF)": round(float(lexical_scores.item(idx)), 4),
                "Semantic (LSA+FT)": round(float(semantic_scores.item(idx)), 4),
                "Document Snippet": doc,
            })

        df = pd.DataFrame(results).sort_values(by="Hybrid Score", ascending=False).head(top_k)
        return df.reset_index(drop=True)
