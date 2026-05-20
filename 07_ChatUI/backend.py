from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv; load_dotenv()
from typing import TypedDict, Annotated
import os


class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages] # ~ We can use operator.add as well, but this more optimized for BaseMessages


def chat_node(state: ChatState):

    prompt = f"You are a helpful assistant, answer the following query in as short as possible: {state['messages']}"
    response = model.invoke(prompt)

    return {'messages': [response]}


llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-R1',
    task='text-generation',
    huggingfacehub_api_token=os.getenv('HF_TOKEN')
)
model = ChatHuggingFace(llm=llm)

checkpointer = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

workflow = graph.compile(checkpointer=checkpointer)