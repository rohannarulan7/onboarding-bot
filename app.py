import streamlit as st
from chatbot import ChatManager, ChatManager2

st.set_page_config(page_title="Onboarding Assistant", layout="wide")

if "manager" not in st.session_state:
    st.session_state.manager = ChatManager2()

st.title("🤖 Onboarding Assistant Chatbot")

# Chat container
chat_placeholder = st.container()

# User input
user_input = st.chat_input("Ask me anything about the project...")

if user_input:
    with st.spinner("Thinking..."):
        answer, memory = st.session_state.manager.ask(user_input)

    with chat_placeholder:
        # Show memory (chat history)
        for turn in memory:
            st.chat_message("user").write(turn["query"])
            st.chat_message("assistant").write(turn["answer"])
