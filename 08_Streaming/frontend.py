import streamlit as st
from backend import workflow
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('> Enter your query here...')

if user_input:

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

#############################################################################################

    initial_state = {'messages': [HumanMessage(content=user_input)]}
    with st.chat_message('assistant'):

        ai_response = st.write_stream(
            # Generator
            chunk.content for chunk, metadata in workflow.stream(
                initial_state, 
                config=CONFIG, 
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_response})

#############################################################################################