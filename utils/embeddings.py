import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_model():
    """
    Loads the sentence embedding model ONCE and caches it for the entire session.
    Without this, the model reloads from disk on every run — which is very slow.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


# Single shared instance used across all modules
model = load_model()
