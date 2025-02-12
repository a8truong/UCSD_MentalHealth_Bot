import streamlit as st
from config.action import retrieve, rag
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

config = RailsConfig.from_path("./config")
rails = LLMRails(config)
rails.register_action(action=retrieve, name="retrieve")
rails.register_action(action=rag, name="rag")
slide_window = 7

def get_chat_history():
#Get the history from the st.session_stage.messages according to the slide window parameter
    
    chat_history = []
    
    start_index = max(0, len(st.session_state.messages) - slide_window)
    for i in range (start_index , len(st.session_state.messages) -1):
         chat_history.append(st.session_state.messages[i])

    return chat_history

def summarize_question_with_history(chat_history, question):
# To get the right context, use the LLM to first summarize the previous conversation
# This will be used to get embeddings and find similar chunks in the docs for context
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [{
            "role": "system",
            "content": f"""
        Based on the chat history below and the question, generate a query that extend the question
        with the chat history provided. The query should be in natural language. 
        Answer with only the query. Do not add any explanation.
        
        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
        """
        }, {
            "role": "user",
            "content": question
        }],
        max_tokens=2048,
        temperature=0
    )

    summary = response.choices[0].message.content 

    return summary

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
                Approach responses in a carefully, being kind and understanding."}
            ]
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        async def chat():
            """
            Generate response asynchronously
            """
            chat_history = get_chat_history()

            if chat_history != []: #There is chat_history, so not first question
                question_summary = summarize_question_with_history(chat_history, prompt)
                response = await rails.generate_async(prompt=question_summary)
            else:
                response = await rails.generate_async(prompt=prompt)
            return response

        response = asyncio.run(chat())

        with st.chat_message("assistant"):
            st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})



if __name__ == "__main__":
    main()