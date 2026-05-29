import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from backend import workflow, ret_all_threads
import os; os.environ['LANGSMITH_PROJECT'] = 'Chatbot'

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

    # This will stream both JSON (tool call) output and chat output
    # with st.chat_message('assistant'):

    #     response = st.write_stream(
    #         chunk.content for chunk, metadata in workflow.stream(
    #             initial_state,
    #             config=CONFIG,
    #             stream_mode='messages'
    #         )
    #         if chunk.content
    #     )

#####################################################################################

    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in workflow.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

#####################################################################################

    st.rerun()