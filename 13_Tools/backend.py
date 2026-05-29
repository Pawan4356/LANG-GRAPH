from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import BaseMessage

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv; load_dotenv()
from typing import TypedDict, Annotated
import os, sqlite3

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):

    prompt = f"You are a helpful assistant, answer the following query in as short as possible: {state['messages']}"
    response = model_with_tools.invoke(prompt)

    return {'messages': [response]}

def ret_all_threads():

    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

search_tool = DuckDuckGoSearchRun()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-R1',
    task='text-generation',
    huggingfacehub_api_token=os.getenv('HF_TOKEN')
)
model = ChatHuggingFace(llm=llm)

######################################################################

tools = [search_tool]
model_with_tools = model.bind_tools(tools=tools)

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
tool_node = ToolNode(tools=tools)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

######################################################################

workflow = graph.compile(checkpointer=checkpointer)