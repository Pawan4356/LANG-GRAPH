import streamlit as st
import uuid
from langchain_core.messages import HumanMessage
from backend import workflow, ret_all_threads

# Utilities

def generate_thread():
    return str(uuid.uuid4())


def add_thread(thread_id):

    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)


def new_chat():

    thread_id = generate_thread()
    st.session_state.thread_id = thread_id
    add_thread(thread_id)
    st.rerun()


def get_chat(thread_id):

    state = workflow.get_state(
        config={'configurable': {'thread_id': thread_id}}
    )

    return state.values.get('messages', [])


def render_chat(messages):

    for message in messages:

        if isinstance(message, HumanMessage):
            role = 'user'
        else:
            role = 'assistant'

        with st.chat_message(role):
            st.markdown(message.content)

# Initialization

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = generate_thread()

if 'chat_threads' not in st.session_state:
    st.session_state.chat_threads = ret_all_threads()

add_thread(st.session_state.thread_id)

# Sidebar 

st.sidebar.title('ChatBot')

if st.sidebar.button('New Chat'):
    new_chat()

st.sidebar.header('Conversations')

for thread_id in reversed(st.session_state.chat_threads):

    if st.sidebar.button(thread_id):
        st.session_state.thread_id = thread_id
        st.rerun()

# Load Current Chat

messages = get_chat(st.session_state.thread_id)
render_chat(messages)

# User Input
 
user_input = st.chat_input('> Enter the query...')

if user_input:

    with st.chat_message('user'):
        st.markdown(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state.thread_id}}
    initial_state = {'messages': [HumanMessage(content=user_input)]}

    with st.chat_message('assistant'):

        response = st.write_stream(
            chunk.content for chunk, metadata in workflow.stream(
                initial_state,
                config=CONFIG,
                stream_mode='messages'
            )
            if chunk.content
        )

    st.rerun()