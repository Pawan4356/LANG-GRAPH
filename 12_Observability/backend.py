from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import BaseMessage, HumanMessage
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv; load_dotenv()
from typing import TypedDict, Annotated
import os
import sqlite3

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages] # ~ We can use operator.add as well, but this more optimized for BaseMessages

def chat_node(state: ChatState):

    prompt = f"You are a helpful assistant, answer the following query in as short as possible: {state['messages']}"
    response = model.invoke(prompt)

    return {'messages': [response]}

def ret_all_threads():

    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-R1',
    task='text-generation',
    huggingfacehub_api_token=os.getenv('HF_TOKEN')
)
model = ChatHuggingFace(llm=llm)

'''
check_same_thread is a boolean parameter used in the sqlite3.connect() function that 
controls whether the connection can be shared across multiple threads.

Key Behavior:- 

    True (Default): The module raises a ProgrammingError if you try to use a database 
    connection in a thread other than the one that created it.

    False: You can share the connection across multiple threads. However, you 
    must manage write serialization yourself (e.g., using a threading lock) to 
    prevent data corruption or transaction errors.
'''
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

workflow = graph.compile(checkpointer=checkpointer)