import streamlit as st
from config.action import rag, concern, semantic_cache
from nemoguardrails import RailsConfig, LLMRails
from dotenv import load_dotenv
import os
import asyncio
import openai
from openai import OpenAI
import re
import spacy

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

config = RailsConfig.from_path("scripts/config")
rails = LLMRails(config)

nlp = spacy.load("en_core_web_sm")

# Initialize cache only once during the session
if "cache" not in st.session_state:
    st.session_state.cache = semantic_cache("4cache.json")

rails.register_action(action=st.session_state.cache.ask, name="ask")
rails.register_action(action=rag, name="rag")
rails.register_action(action=concern, name="concern")
slide_window = 2  # Define when to summarize chat history

def get_chat_history():
    """
    Retrieve the last 5 messages from chat history (excluding the most recent one).
    """
    if len(st.session_state.messages) <= 2:
        return []  # Not enough messages to summarize
      
    # Exclude the last two messages (latest user query and assistant response)
    return st.session_state.messages[-5:-1]

def extract_keywords(chat_history):
    """Extracts meaningful keywords from the chat history using spaCy."""
    if not chat_history:
        return []

    text = " ".join([msg["content"] for msg in chat_history if msg["role"] == "user"])
    doc = nlp(text)

    keywords = set()
    for token in doc:
        if token.pos_ in {"NOUN", "VERB", "PROPN"} and token.is_alpha:  # Extract nouns, verbs, and proper nouns
            keywords.add(token.lemma_.lower())  # Use lemma to normalize words

    return list(keywords)

def generate_contextual_prompt(prompt, chat_history):
    """Enhances the prompt with relevant keywords from chat history."""
    keywords = extract_keywords(chat_history)
    if keywords:
        keyword_str = ", ".join(keywords)
        prompt = f"Prior discussion included: {keyword_str}. User now asks: {prompt}"
    
    return prompt

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

            If the last sentence is a question, keep the question in the summary.
            
            <chat_history>
            {chat_history}
            </chat_history>
            """
        }],
        max_tokens=512,
        temperature=0
    )

    return response.choices[0].message.content.strip()

# async def chat(prompt):
#     """
#     Generate a response asynchronously, adding context from chat history.
#     """
#     chat_history = get_chat_history()
    
#     if len(chat_history) >= slide_window:  # Summarize history if needed
#         summarized_history = " ".join([msg["content"] for msg in chat_history])
#         prompt = f"Chat history summary: {summarized_history}. New question: {prompt}"
    
#     prompt = generate_contextual_prompt(prompt, chat_history)  # Add keywords from history
#     response = await rails.generate_async(prompt=prompt)
#     print(f"Final Prompt: {prompt}")

#     return response

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
                print(chat_history)
                summarized_history = summarize_chat_history(chat_history)
                p = "Chat history summary: " + summarized_history + " New question: " + prompt
                response = await rails.generate_async(prompt=p)
                print(p)
            else:
                response = await rails.generate_async(prompt=prompt)
                print(">Prompt: " + prompt)

            return response

        response = asyncio.run(chat())

        match = re.search(r'Bot message:\s*"([^"]+)"', response)
        if match:
            response = match.group(1)  # Extracted message

        with st.chat_message("assistant"):
            st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
