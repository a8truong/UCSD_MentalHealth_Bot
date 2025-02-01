import streamlit as st
from rag import initialize_rag

st.title("UCSD Mental Health Bot")

# Initialize RAG chain
rag_chain = initialize_rag()

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

    # Generate RAG response
    response = rag_chain.invoke({"question": prompt})

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})