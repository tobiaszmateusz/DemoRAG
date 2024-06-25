__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
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




def setup_chain(llm=None, memory=None, inject_knowledge=None, system_message=None, context_prompt=None, retriever=None):
    global view
    if not inject_knowledge:
        if view.use_memory_of_conversation:
        # Custom conversational chain
            return ConversationalChain(
                llm=llm,
                memory=memory,
                system_message=system_message,
                verbose=True)
        else:
            return ConversationalChain(
                llm=llm,
                system_message=system_message,
                verbose=True)
    else:
        if view.use_memory_of_conversation:
            return ConversationalRetrievalChain(
                llm=llm,
                retriever=retriever,
                memory=memory,
                system_message=system_message,
                context_prompt=context_prompt,
                verbose=True)
        else:
            return ConversationalRetrievalChain(
                llm=llm,
                retriever=retriever,
                system_message=system_message,
                context_prompt=context_prompt,
                verbose=True)


def setup_memory():
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    return ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)


def ask():
    global view
    global memory

    model = LLM(model=view.model_name,
                temperature=view.temperature,
                tech=view.technology,
                top_p=view.top_p,
                frequency_penalty=view.frequency_penalty,
                presence_penalty=view.presence_penalty).llm

    if (view.use_memory_of_conversation):
        chain = setup_chain(llm=model,
                            memory=memory,
                            retriever=chroma_database.retriever,
                            inject_knowledge=True,
                            system_message=view.system_message,
                            context_prompt=view.context_prompt)
    else:
        chain = setup_chain(llm=model,
                            retriever=chroma_database.retriever,
                            inject_knowledge=True,
                            system_message=view.system_message,
                            context_prompt=view.context_prompt)

    return chain



def main():
    load_dotenv()

    global view
    global memory
    global chroma_database


    memory = setup_memory()
    view = StreamlitChatView(memory=memory)
    chroma_database = ChromaDatabase(view)


    if view.embeddings_model_name is not None:
        if chroma_database.check_database():
            chroma_database.get_client()
            st.button("Add to ChromaDB",
                  on_click=lambda: chroma_database.get_or_create_collection_mine())
    else:
        st.write("Embedding not selected - you are not able to load documents to database")
        return

    st.session_state["ingestion_spinner"] = st.empty()

    chroma_database.get_collections()
    if len(chroma_database.collections) > 0:
        for doc in chroma_database.collections:
            with st.container():
                cols = st.columns([2, 1])  # Create two columns
                with cols[0]:  # Right column
                    st.checkbox(doc.name, key=doc.name, on_change=chroma_database.check_selected, args=(doc.name,))

        st.button("Create retriever from selected document", on_click=lambda: chroma_database.create_retriever_from_selected_documents())

    if chroma_database.retriever is not None:
        # Display previous messages
        for message in memory.chat_memory.messages:
            view.add_message(message.content, 'assistant' if message.type == 'ai' else 'user')

        if view.user_query:
            view.add_message(view.user_query, "user")
            response = ask().run({"question": view.user_query})
            view.add_message(response, "assistant")

    else:
        st.info(
            "LOAD DOCUMENTS OR SELECT EXISTING AND CREATE RETRIEVER",
            icon="ℹ️")


if __name__ == "__main__":
    main()