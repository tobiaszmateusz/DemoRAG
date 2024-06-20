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


def check_database():
    embedding_name = st.session_state["embedding"]
    tech = st.session_state["tech"]
    create_database_method = st.session_state["create_database_method"]
    persist_directory = f"./bases/chroma_db_{tech}_{embedding_name.replace(':', '_')}_{create_database_method}"
    return os.path.isdir(persist_directory) is not None


def create_database():
    embedding_name = st.session_state["embedding"]
    tech = st.session_state["tech"]
    create_database_method = st.session_state["create_database_method"]
    persist_directory = f"./bases/chroma_db_{tech}_{embedding_name.replace(':', '_')}_{create_database_method}"
    chromadb.PersistentClient(path=persist_directory)


def get_collections():
    embedding_name = st.session_state["embedding"]
    tech = st.session_state["tech"]
    create_database_method = st.session_state["create_database_method"]
    persist_directory = f"./bases/chroma_db_{tech}_{embedding_name.replace(':', '_')}_{create_database_method}"
    if os.path.isdir(persist_directory):
        client = chromadb.PersistentClient(path=persist_directory)
        st.session_state["collections"] = [collection.name for collection in client.list_collections()]


def add_to_db_if_not_exist(path: str, name: str):
    if st.session_state["create_database_method"] == "UNSTRUCTURED":
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
        model = LLM(model=st.session_state["model"], temperature=0.5, tech=st.session_state["tech"]).llm
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        text_summaries = texts  # Skip it
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
        summary = text_summaries + table_summaries

    if st.session_state["create_database_method"] == "DOCUMENT SPLITTING":
        docs = PyPDFLoader(path).load()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=800, chunk_overlap=100
        )
        temp_summary = text_splitter.split_documents(docs)
        summary = [doc.page_content for doc in temp_summary]

    ids = [str(uuid.uuid4()) for _ in summary]
    documents = [Document(page_content=s, metadata={"file": name}) for s in summary]
    embedding_name = st.session_state["embedding"]
    create_database_method = st.session_state["create_database_method"]
    tech = st.session_state["tech"]
    persist_directory = f"./bases/chroma_db_{tech}_{embedding_name.replace(':', '_')}_{create_database_method}"
    vectorstore = Chroma.from_documents(
        documents=documents,
        ids=ids,
        collection_name=name,
        embedding=EMBEDDING(model=st.session_state["embedding"], tech=st.session_state["tech"]).embedding,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    # st.session_state["retrievers"].append({"db": vectorstore.as_retriever(), "name": name, "selected": False})


def create_retriever_from_selected_documents():
    embedding_name = st.session_state["embedding"]
    tech = st.session_state["tech"]
    create_database_method = st.session_state["create_database_method"]
    persist_directory = f"./bases/chroma_db_{tech}_{embedding_name.replace(':', '_')}_{create_database_method}"
    if os.path.isdir(persist_directory):
        for name in st.session_state["selected_documents"]:
            db = Chroma(persist_directory=persist_directory, collection_name=name,
                        embedding_function=EMBEDDING(model=st.session_state["embedding"],
                                                     tech=st.session_state["tech"]).embedding)
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


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def ask(query):
    template = """Answer the question based only on the following context, which can include text and tables:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = LLM(model=st.session_state["model"], tech=st.session_state["tech"]).llm

    chain = (
            {"context": st.session_state["retriever"], "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )
    output = chain.invoke(query)
    if st.session_state["tech"] == "HUGGINGFACE":
        answer_index = output.find("Answer:")
        if answer_index != -1:
            return output[answer_index + 7:]
    else:
        return output


def restart_assistant():
    st.session_state["collections"] = None
    st.session_state["selected_documents"] = None
    st.session_state["retriever"] = None
    st.session_state["retrievers"] = None
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""
    # st.rerun()


def display_messages():
    st.subheader("Chat")
    if "messages" in st.session_state:
        for i, (msg, is_user) in enumerate(st.session_state["messages"]):
            message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def main():
    load_dotenv()
    st.set_page_config(
        page_title="PDF AI",
    )
    st.title("PDF Assistant")

    tech = get_technology_work()
    if "tech" not in st.session_state:
        st.session_state["tech"] = tech
    # Restart the assistant if assistant_type has changed
    elif st.session_state["tech"] != tech:
        st.session_state["tech"] = tech

    model = get_model_work()
    if "model" not in st.session_state:
        st.session_state["model"] = model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["model"] != model:
        st.session_state["model"] = model

    create_database_method = get_create_database_method()
    if "create_database_method" not in st.session_state:
        st.session_state["create_database_method"] = create_database_method
    # Restart the assistant if assistant_type has changed
    elif st.session_state["create_database_method"] != create_database_method:
        st.session_state["create_database_method"] = create_database_method
        restart_assistant()

    embedding = get_embed_model_work()
    if "embedding" not in st.session_state:
        st.session_state["embedding"] = embedding
    # Restart the assistant if assistant_type has changed
    elif st.session_state["embedding"] != embedding:
        st.session_state["embedding"] = embedding
        restart_assistant()

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
    st.info("OLLAMA w Technology nie działa", icon="ℹ️")
    st.info("UNTRUCTURED w DatabaseModel nie działa", icon="ℹ️")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        # on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    if st.session_state["embedding"] is not None:
        if not check_database():
            # get_collections()
        # else:
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
        display_messages()
        st.text_input("Message", key="user_input")
        st.button("Generate answer", on_click=lambda: process_input())

    else:
        st.info(
            "LOAD DOCUMENTS OR SELECT EXISTING AND CREATE RETRIEVER",
            icon="ℹ️")


if __name__ == "__main__":
    main()