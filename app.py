import streamlit as st
from config.action import retrieve, rag
from nemoguardrails import RailsConfig, LLMRails
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

config = RailsConfig.from_path("./config")
rails = LLMRails(config)
rails.register_action(action=retrieve, name="retrieve")
rails.register_action(action=rag, name="rag")


st.title("UCSD Mental Health Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("How can I help you?"):

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a therapist chatbot focused on UCSD students. \
             Approach responses in a carefully, being kind and understanding."}
        ]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    async def chat():
        """
        Generate response asynchronously
        """
        response = await rails.generate_async(prompt=prompt)
        return response

    response = asyncio.run(chat())

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})