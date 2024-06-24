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


def reset_mess(memory):
    memory.clear()

def reset_assistant():
    st.session_state["collections"] = None
    st.session_state["selected_documents"] = None
    st.session_state["retriever"] = None
    st.session_state["retrievers"] = None
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""


class StreamlitChatView:
    def __init__(self, memory) -> None:
        self.memory = memory
        st.set_page_config(page_title="PDF AI", page_icon="📚", layout="wide")
        st.title("PDF Assistant")
        with st.sidebar:
            st.title("RAG ChatGPT")
            with st.expander("Model parameters"):
                self.technology = get_technology_work(reset_assistant=reset_assistant)
                self.model_name = get_model_work(key=None, technology=self.technology)
                self.temperature = st.slider("Temperature", min_value=0., max_value=2., value=0.7, step=0.01)
                self.top_p = st.slider("Top p", min_value=0., max_value=1., value=1., step=0.01)
                self.frequency_penalty = st.slider("Frequency penalty", min_value=0., max_value=2., value=0., step=0.01)
                self.presence_penalty = st.slider("Presence penalty", min_value=0., max_value=2., value=0., step=0.01)
            with st.expander("Prompts"):
                curdate = datetime.datetime.now().strftime("%Y-%m-%d")
                model_name = self.model_name.replace('-turbo', '').upper()
                system_message = (f"You are ChatGPT, a large language model trained by OpenAI, "
                                  f"based on the {model_name} architecture.\n"
                                  f"Knowledge cutoff: 2021-09\n"
                                  f"Current date: {curdate}\n")
                self.system_message = st.text_area("System message", value=system_message)
                self.context_prompt = st.text_area("Context prompt", value=TEMPLATE)
            with st.expander("Database syle creation"):
                self.create_database_method = get_create_database_method()
            with st.expander("Embeddings parameters"):
                self.embeddings_model_name = get_embed_model_work(key=None, technology=self.technology, reset_assistant=reset_assistant)
            self.use_memory_of_conversation = st.checkbox("Use previous chats messages", value=True, on_change=lambda: reset_mess(self.memory))
            
        persist_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bases", f"chroma_db_{self.technology}_{self.embeddings_model_name.replace(':', '_')}_{self.create_database_method}")
        self.user_query = st.chat_input(placeholder="Ask me anything!")

    def add_message(self, message: str, author: str):
        assert author in ["user", "assistant"]
        with st.chat_message(author):
            formatted_response = message.replace('$', '\$')
            st.markdown(formatted_response)

    def add_message_stream(self, author: str):
        assert author in ["user", "assistant"]
        return StreamHandler(st.chat_message(author).empty())

