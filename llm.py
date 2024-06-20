from langchain.chat_models import ChatOllama, ChatOpenAI
from langchain_groq.chat_models import ChatGroq
from langchain.embeddings import OllamaEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings

from langchain_community.llms import HuggingFaceHub


class LLM:
    def __init__(self, model, temperature=0, format=None, tech=None):
        self.model = model
        self.temperature = temperature
        self.format = format
        self.tech = tech
        if tech == 'OPENAI':
            self.llm = ChatOpenAI(model=model, temperature=temperature)
        elif tech == 'OLLAMA':
            self.llm = ChatOllama(model=model, temperature=temperature)
        elif tech == 'GROQ':
            self.llm = ChatGroq(model=model, temperature=temperature)
        elif tech == 'HUGGINGFACE':
            self.llm = HuggingFaceHub(repo_id=model,  model_kwargs={"temperature":0.1, "max_length":500})


class EMBEDDING:
    def __init__(self, model, temperature=0, tech=None, dimensions=1024):
        self.model = model
        self.temperature = temperature
        self.format = format
        self.tech = tech
        self.dimensions = dimensions
        if tech == 'OPENAI':
            self.embedding = OpenAIEmbeddings(model=model)
        elif tech == 'OLLAMA':
            self.embedding = OllamaEmbeddings(model=model)
        elif tech == 'GROQ':
            self.embedding = OpenAIEmbeddings(model=model)
        elif tech == 'HUGGINGFACE':
            self.embedding = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={'device':'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
