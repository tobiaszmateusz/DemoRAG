from typing import Optional
from os import getenv, environ
import json
try:
    import streamlit as st
except ImportError:
    raise ImportError("`streamlit` library not installed. Please install using `pip install streamlit`")

def get_openai_key_sidebar() -> Optional[str]:
    openai_key: Optional[str] = getenv("OPENAI_API_KEY")
    if openai_key is None or openai_key == "" or openai_key == "sk-***":
        api_key = st.sidebar.text_input("OpenAI API key", placeholder="sk-***", key="api_key")
        if api_key != "sk-***" or api_key != "" or api_key is not None:
            openai_key = api_key

    if openai_key is not None and openai_key != "":
        st.session_state["OPENAI_API_KEY"] = openai_key
        environ["OPENAI_API_KEY"] = openai_key

    return openai_key


def get_huggingface_key_sidebar() -> Optional[str]:
    huggingface_key: Optional[str] = getenv("HUGGINGFACEAPP_API_TOKEN")
    if huggingface_key is None or huggingface_key == "" or huggingface_key == "hf_***":
        api_key_huggingface = st.sidebar.text_input("HuggingFace API key", placeholder="hf_***", key="api_key")
        if api_key_huggingface != "hf_***" or api_key_huggingface != "" or api_key_huggingface is not None:
            huggingface_key = api_key_huggingface

    if huggingface_key is not None and huggingface_key != "":
        st.session_state["HUGGINGFACEAPP_API_TOKEN"] = huggingface_key
        environ["HUGGINGFACEAPP_API_TOKEN"] = huggingface_key

    huggingface_key2: Optional[str] = getenv("HUGGINGFACEHUB_API_TOKEN")
    if huggingface_key2 is None or huggingface_key2 == "" or huggingface_key2 == "hf_***":
        api_key_huggingface2 = st.sidebar.text_input("HuggingFaceHub API key", placeholder="hf_***", key="api_key2")
        if api_key_huggingface2 != "hf_***" or api_key_huggingface2 != "" or api_key_huggingface2 is not None:
            huggingface_key2 = api_key_huggingface2

    if huggingface_key2 is not None and huggingface_key2 != "":
        st.session_state["HUGGINGFACEHUB_API_TOKEN"] = huggingface_key2
        environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_key2

    return huggingface_key


def get_groq_key_sidebar() -> Optional[str]:
    groq_key: Optional[str] = getenv("GROQ_API_KEY")
    if groq_key is None or groq_key == "" or groq_key == "gsk_***":
        api_key_groq = st.sidebar.text_input("Groq API key", placeholder="gsk_***", key="api_key")
        if api_key_groq != "gsk_***" or api_key_groq != "" or api_key_groq is not None:
            groq_key = api_key_groq

    if groq_key is not None and groq_key != "":
        st.session_state["GROQ_API_KEY"] = groq_key
        environ["GROQ_API_KEY"] = groq_key

    return groq_key

def get_ollama_key_sidebar() -> Optional[str]:
    groq_key: Optional[str] = getenv("GROQ_API_KEY")
    if groq_key is None or groq_key == "" or groq_key == "gsk_***":
        api_key_groq = st.sidebar.text_input("Groq API key", placeholder="gsk_***", key="api_key")
        if api_key_groq != "gsk_***" or api_key_groq != "" or api_key_groq is not None:
            groq_key = api_key_groq

    if groq_key is not None and groq_key != "":
        st.session_state["GROQ_API_KEY"] = groq_key
        environ["GROQ_API_KEY"] = groq_key

    return groq_key


def get_technology_work() -> Optional[str]:
    with open('langchain_rag_v2/technology.json', 'r') as f:
        TECHNOLOGY = json.load(f)
    technology_options = list(TECHNOLOGY.keys())
    technology_functions = {
        "OPENAI": get_openai_key_sidebar,
        "HUGGINGFACE": get_huggingface_key_sidebar,
        "GROQ": get_groq_key_sidebar,
    }

    technology_work = st.sidebar.selectbox("Technology", technology_options)

    if technology_work and technology_work != "OLLAMA":
        technology_functions.get(technology_work, lambda: None)()

    return technology_work


def get_model_work(
    key: Optional[str] = None
) -> Optional[str]:
    with open('langchain_rag_v2/technology.json', 'r') as f:
        TECHNOLOGY = json.load(f)
    model_keys = list(TECHNOLOGY[st.session_state["tech"]]["model"])
    if key is not None:
        return st.sidebar.selectbox(f"Model {key}:", model_keys, key=key)
    else:
        return st.sidebar.selectbox("Model:", model_keys)



def get_embed_model_work() -> Optional[str]:
    with open('langchain_rag_v2/technology.json', 'r') as f:
        TECHNOLOGY = json.load(f)
    embeddings_keys = list(TECHNOLOGY[st.session_state["tech"]]["embeddings"])
    embeddings_work = st.sidebar.selectbox("Embeddings:", embeddings_keys)
    return embeddings_work


def get_create_database_method():
    with open('langchain_rag_v2/technology.json', 'r') as f:
        TECHNOLOGY = json.load(f)
    create_database_method = TECHNOLOGY["create_database_method"]
    create_database_method_work = st.sidebar.selectbox("Database Method:", create_database_method)
    return create_database_method_work


