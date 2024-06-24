from langchain.chat_models import ChatOllama, ChatOpenAI
from langchain_groq.chat_models import ChatGroq
from langchain.embeddings import OllamaEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings

from langchain_community.llms import HuggingFaceHub


class LLM:
    def __init__(self, model, temperature=0, format=None, tech=None, top_p=None, frequency_penalty=None, presence_penalty=None):
        self.model = model
        self.temperature = temperature
        self.format = format
        self.tech = tech
        if tech == 'OPENAI':
            self.llm = ChatOpenAI(
                model_name=model,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty)
        elif tech == 'OLLAMA':
            self.llm = ChatOllama(
                model_name=model,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty)
        elif tech == 'GROQ':
            self.llm = ChatGroq(
                model_name=model,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty)
        elif tech == 'HUGGINGFACE':
            self.llm = HuggingFaceHub(repo_id=model,
                                      model_kwargs={"temperature":{temperature},
                                                    "max_length":500,
                                                    'top_p':{top_p},
                                                    'frequency_penalty':{frequency_penalty},
                                                    'presence_penalty':{presence_penalty}})


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
