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


def check_database():
    global view
    return os.path.isdir(view.persist_directory) is not None


def create_database():
    global view
    chromadb.PersistentClient(path=view.persist_directory)


def get_collections():
    global view
    if os.path.isdir(view.persist_directory):
        client = chromadb.PersistentClient(path=view.persist_directory)
        st.session_state["collections"] = [collection.name for collection in client.list_collections()]


def add_to_db_if_not_exist(path: str, name: str):
    global view
    if view.create_database_method == "UNSTRUCTURED":
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
        model = LLM(model=view.model_name,
                    temperature=view.temperature,
                    tech=view.technology,
                    top_p=view.top_p,
                    frequency_penalty=view.frequency_penalty,
                    presence_penalty=view.presence_penalty).llm
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        text_summaries = texts  # Skip it
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
        summary = text_summaries + table_summaries

    if view.create_database_method  == "DOCUMENT SPLITTING":
        docs = PyPDFLoader(path).load()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=800, chunk_overlap=100
        )
        temp_summary = text_splitter.split_documents(docs)
        summary = [doc.page_content for doc in temp_summary]

    ids = [str(uuid.uuid4()) for _ in summary]
    documents = [Document(page_content=s, metadata={"file": name}) for s in summary]
    vectorstore = Chroma.from_documents(
        documents=documents,
        ids=ids,
        collection_name=name,
        embedding=EMBEDDING(model=view.embeddings_model_name, tech=view.technology).embedding,
        persist_directory=view.persist_directory
    )
    vectorstore.persist()
    # st.session_state["retrievers"].append({"db": vectorstore.as_retriever(), "name": name, "selected": False})


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


def create_retriever_from_selected_documents():
    global view
    if os.path.isdir(view.persist_directory):
        for name in st.session_state["selected_documents"]:
            db = Chroma(persist_directory=view.persist_directory, collection_name=name,
                        embedding_function=EMBEDDING(model=view.embeddings_model_name,
                                                     tech=view.technology).embedding)
            if any(i['file'] == name for i in db.get()['metadatas']):
                # st.session_state["retrievers"].append({"db": db.as_retriever(), "name": name, "selected": False})
                if st.session_state["retrievers"] is None:
                    st.session_state["retrievers"] = []
                st.session_state["retrievers"].append(db.as_retriever())
    if len(st.session_state["retrievers"]) > 0:
        st.session_state["retriever"] = MergerRetriever(retrievers=st.session_state["retrievers"])


def check_existing():
    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name
        if st.session_state["collections"] is None or file_path not in st.session_state["collections"]:
            with st.session_state["ingestion_spinner"], st.spinner(f"Adding {file.name}"):
                add_to_db_if_not_exist(file_path, file.name)


def check_selected(name):
    if "selected_documents" not in st.session_state or st.session_state["selected_documents"] == None:
        st.session_state["selected_documents"] = []
    selected_documents = st.session_state.get("selected_documents", [])
    if name in selected_documents:
        selected_documents.remove(name)
    else:
        selected_documents.append(name)
    st.session_state.selected_documents = selected_documents


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
                            retriever=st.session_state["retriever"],
                            inject_knowledge=True,
                            system_message=view.system_message,
                            context_prompt=view.context_prompt)
    else:
        chain = setup_chain(llm=model,
                            retriever=st.session_state["retriever"],
                            inject_knowledge=True,
                            system_message=view.system_message,
                            context_prompt=view.context_prompt)

    return chain


def restart_assistant():
    st.session_state["collections"] = None
    st.session_state["selected_documents"] = None
    st.session_state["retriever"] = None
    st.session_state["retrievers"] = None
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""
    # st.rerun()


view = None

def main():
    load_dotenv()

    global view
    global memory

    memory = setup_memory()
    view = StreamlitChatView(memory=memory)


    if "collections" not in st.session_state:
        st.session_state["collections"] = None

    if "selected_documents" not in st.session_state:
        st.session_state["selected_documents"] = None

    if "retrievers" not in st.session_state:
        st.session_state["retrievers"] = None

    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""

    st.info("Database method to rozróżnienie bazy danych na taką, która tylko dzieli dokument pdf oraz na taką która przez inny model czyta dokuemnt pdf", icon="ℹ️")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        # on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    if view.embeddings_model_name is not None:
        if not check_database():
            create_database()
        st.button("Add to ChromaDB",
                  on_click=lambda: check_existing())
    else:
        st.write("Embedding not selected - you are not able to load documents to database")
        return

    st.session_state["ingestion_spinner"] = st.empty()

    get_collections()

    if st.session_state["collections"] is not None and len(st.session_state["collections"]) > 0:
        st.write("Select documents:")
        for doc in st.session_state["collections"]:
            with st.container():
                cols = st.columns([2, 1])  # Create two columns
                with cols[0]:  # Right column
                    st.checkbox(doc, key=doc, on_change=check_selected, args=(doc,))

        st.button("Create retriever from selected document", on_click=lambda: create_retriever_from_selected_documents())

    if "retriever" in st.session_state and st.session_state["retriever"] is not None:
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