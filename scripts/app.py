import streamlit as st
from config.action import retrieve, rag, concern
from nemoguardrails import RailsConfig, LLMRails
from dotenv import load_dotenv
import os
import asyncio
import openai
from openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

config = RailsConfig.from_path("./scripts/config")
rails = LLMRails(config)
rails.register_action(action=retrieve, name="retrieve")
rails.register_action(action=rag, name="rag")
rails.register_action(action=concern, name="concern")
slide_window = 2  # Define when to summarize chat history

def get_chat_history():
    """
    Get the chat history from st.session_state.messages, excluding the last two messages.
    """
    if len(st.session_state.messages) <= 2:
        return []  # Not enough messages to summarize

    # Exclude the last two messages (latest user query and assistant response)
    return st.session_state.messages[:-2][-slide_window:]

def summarize_chat_history(chat_history):
    """
    Summarizes the chat history, excluding the most recent two messages.
    """
    if not chat_history:
        return ""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": f"""
            Summarize the chat history provided below into a concise and natural form, 
            keeping the key details relevant to the conversation. If mental health concerns 
            such as stress, anxiety, or emotional distress are mentioned, ensure the summary 
            includes references to helpful resources like mental health workshops, events, or counseling services.
            
            <chat_history>
            {chat_history}
            </chat_history>
            """
        }],
        max_tokens=512,
        temperature=0
    )

    return response.choices[0].message.content.strip()

def main():
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
                Approach responses carefully, being kind and understanding. Make it interactive."}
            ]
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        async def chat():
            """
            Generate response asynchronously, summarizing chat history if needed.
            """
            chat_history = get_chat_history()

            if len(chat_history) >= slide_window:  # Summarize history if it exceeds threshold
                summarized_history = summarize_chat_history(chat_history)
                response = await rails.generate_async(prompt=summarized_history + " " + prompt)
                print(summarized_history)
            else:
                response = await rails.generate_async(prompt=prompt)

            return response

        response = asyncio.run(chat())

        with st.chat_message("assistant"):
            st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
