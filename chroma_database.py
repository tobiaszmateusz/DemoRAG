import datetime
import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import (ConversationBufferMemory,
                              StreamlitChatMessageHistory)

from chains.conversational_chain import ConversationalChain
from chains.conversational_retrieval_chain import (
    TEMPLATE, ConversationalRetrievalChain)
from utils.streaming import StreamHandler
from utils.available_gpt_models import get_available_openai_models
from extras import *
import chromadb
from langgraph.graph import END, StateGraph
from pprint import pprint
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
from extras import *
import tempfile
import os
from langchain.vectorstores import Chroma
import chromadb
from unstructured.partition.pdf import partition_pdf
from langchain.prompts import ChatPromptTemplate
from llm import LLM, EMBEDDING
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.document import Document
import uuid
from langchain.retrievers import MergerRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chat import *
from chroma_database import *
from chromadb.utils import embedding_functions


class ChromaDatabase:
    def __init__(self, view) -> None:
        self.view = view
        self.client = None
        self.collections = []
        self.collection = None
        self.selected_documents = []
        self.retrievers = []
        self.retriever = None


    def get_client(self):
        self.client = chromadb.PersistentClient(path=self.view.persist_directory)


    def get_collections(self):
        self.collections = self.client.list_collections()

    def get_or_create_collection_mine(self):
        for file in st.session_state["file_uploader"]:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(file.getbuffer())
                file_path = tf.name
                with st.session_state["ingestion_spinner"], st.spinner(f"Adding {file.name}"):
                    self.collection = self.client.get_or_create_collection(file.name)
                    if self.collection.embeddings is None:
                        self.create_embeddings(file_path, file.name)



    def check_database(self):
        return os.path.isdir(self.view.persist_directory) is not None



    def create_embeddings(self, path: str, name: str):
        if self.view.create_database_method == "UNSTRUCTURED":
            raw_pdf_elements = partition_pdf(
                filename=path,
                extract_images_in_pdf=False,
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=2000,
                new_after_n_chars=1800,
                combine_text_under_n_chars=1000,
            )

            tables = [str(element) for element in raw_pdf_elements if
                      "unstructured.documents.elements.Table" in str(type(element))]
            texts = [str(element) for element in raw_pdf_elements if
                     "unstructured.documents.elements.CompositeElement" in str(type(element))]
            prompt_text = """You are an assistant tasked with summarizing tables and text. Give a concise summary of the table or text. Table or text chunk: {element} """
            prompt = ChatPromptTemplate.from_template(prompt_text)
            model = LLM(model=self.view.model_name,
                        temperature=self.view.temperature,
                        tech=self.view.technology,
                        top_p=self.view.top_p,
                        frequency_penalty=self.view.frequency_penalty,
                        presence_penalty=self.view.presence_penalty).llm
            summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
            text_summaries = texts  # Skip it
            table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
            summary = text_summaries + table_summaries

        if self.view.create_database_method == "DOCUMENT SPLITTING":
            docs = PyPDFLoader(path).load()
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=800, chunk_overlap=100
            )
            temp_summary = text_splitter.split_documents(docs)
            summary = [doc.page_content for doc in temp_summary]

        # ids = [str(uuid.uuid4()) for _ in summary]
        # documents = [Document(page_content=s, metadata={"file": name}) for s in summary]
        # embeddings = self.openai_ef(documents)
        # self.collection.add(
        #     embeddings=embeddings,
        #     documents=documents,
        #     ids=ids
        # )
        ids = [str(uuid.uuid4()) for _ in summary]
        documents = [Document(page_content=s, metadata={"file": name}) for s in summary]
        Chroma.from_documents(
            documents=documents,
            ids=ids,
            collection_name=name,
            embedding=EMBEDDING(model=self.view.embeddings_model_name, tech=self.view.technology).embedding,
            persist_directory=self.view.persist_directory
        )

    def create_retriever_from_selected_documents(self):
        self.retrievers.clear()
        if len(self.selected_documents) > 0:
            for name in self.selected_documents:
                db = Chroma(persist_directory=self.view.persist_directory, collection_name=name,
                            embedding_function=EMBEDDING(model=self.view.embeddings_model_name,
                                                         tech=self.view.technology).embedding)
                if any(i['file'] == name for i in db.get()['metadatas']):
                    # st.session_state["retrievers"].append({"db": db.as_retriever(), "name": name, "selected": False})
                    st.session_state["retrievers"].append(db.as_retriever())
        if len(self.retrievers) > 0:
            self.retriever = MergerRetriever(retrievers=self.retrievers)


    def check_selected(self, name):
        if name in self.selected_documents:
            self.selected_documents.remove(name)
        else:
            self.elected_documents.append(name)

